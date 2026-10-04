import sys
import os
import json
import faulthandler
import signal
from datetime import timedelta
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.strategies import (
    DDPStrategy,
    FSDPStrategy,
)
from omegaconf import OmegaConf
import torch
import logging

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger, Logger
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.plugins.precision import FSDPPrecision

from ifmdock.data.parse.base import save_config
from ifmdock.data.modules.training import setup_training_datamodule
from ifmdock.models.pl_modules import setup_model
from ifmdock.models.layers.tensor_product import TensorProductConvLayer

from ifmdock.utils.callbacks import setup_callbacks
from ifmdock import is_compact_weights_checkpoint


def load_initial_model_weights(model, checkpoint_path):
    """Load compatible weights without resuming optimizer/epoch state.

    Stage-2 adds the EC-Dock distance branch, so a stage-1 Lightning checkpoint
    cannot be passed as ``ckpt_path`` (that would require all new keys).  This
    intentionally seeds only matching parameters and starts a fresh stage-2
    optimizer and epoch counter.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("ema_weights", checkpoint.get("state_dict", checkpoint))
    # IFMDock's EMA tracks the inner score network and therefore omits the
    # Lightning module's ``model.`` prefix used by ``state_dict``.
    if state_dict and not next(iter(state_dict)).startswith("model."):
        state_dict = {f"model.{key}": value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    rank_zero_info(
        "Loaded stage-2 initial weights from %s (missing=%d, unexpected=%d)"
        % (checkpoint_path, len(incompatible.missing_keys), len(incompatible.unexpected_keys))
    )
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if checkpoint_epoch is None:
        return None
    # Full Lightning checkpoints store a zero-based loop epoch.  Our compact
    # ``ifmdock_weights_only_v1`` artifact stores the already one-based epoch
    # written by ModelWeightsCallback.  Treating both alike skipped epoch 82
    # when branching from the archived epoch-81 EMA weights.
    if is_compact_weights_checkpoint(checkpoint):
        return int(checkpoint_epoch)
    return int(checkpoint_epoch) + 1


def set_weight_only_start_epoch(trainer, start_epoch):
    """Start Lightning's epoch counter after a weights-only checkpoint.

    A weights-only restart deliberately does not restore optimizer state, but
    it should still retain the run's global epoch numbering.  Lightning has no
    public ``initial_epoch`` argument, so initialize all fit-loop progress
    counters consistently before ``fit`` starts.
    """
    if start_epoch <= 0:
        return
    progress = trainer.fit_loop.epoch_progress
    for tracker in (progress.current, progress.total):
        tracker.ready = start_epoch
        tracker.started = start_epoch
        tracker.processed = start_epoch
        tracker.completed = start_epoch
    rank_zero_info(f"Continuing global epoch count from {start_epoch}")


def setup_strategy(cfg):
    strategy_str = cfg.type.lower()  # make it lowercase

    if strategy_str == "auto":
        rank_zero_info(
            "INFO: Strategy automatically selected by lightning: pl_strategy='auto'"
        )
        pl_strategy = "auto"
        return pl_strategy

    if not torch.cuda.is_available():
        return DDPStrategy(find_unused_parameters=True)

    SHARDING_STRATEGY = {
        "full": "FULL_SHARD",
        "hybrid": "HYBRID_SHARD",
        "none": "NO_SHARD",
        "grad": "SHARD_GRAD_OP",
    }

    strategy_kwargs = {}
    if strategy_str == "ddp":
        # Stage-1 docking has conditional paths (notably ligands with no
        # rotatable bond).  The active autograd graph can therefore vary
        # between ranks and batches; static-graph DDP eventually stalls in
        # its reducer under this workload.  Have DDP discover unused
        # parameters every step instead.
        rank_zero_info(
            "DDP: static_graph=False, find_unused_parameters=%s, "
            "broadcast_buffers=False, init_sync=%s"
            % (
                cfg.get("find_unused_parameters", True),
                cfg.get("init_sync", True),
            )
        )
        return DDPStrategy(
            static_graph=False,
            find_unused_parameters=cfg.get("find_unused_parameters", True),
            process_group_backend="nccl",
            # Match EC-Dock's conservative DDP setup.  In particular, do not
            # alias parameter gradients to reducer buckets: IFMDock's custom
            # optimisation/EMA hooks have repeatedly stalled with that memory
            # optimisation enabled on variable-size molecular graphs.
            broadcast_buffers=False,
            # Keep the standard parameter/buffer verification broadcast.  On
            # this host P2P is disabled in the launcher because RTX 5090 peer
            # access is unsupported; with SHM transport, init synchronization
            # is stable and prevents ranks from entering collectives with
            # diverging reducer state.
            init_sync=cfg.get("init_sync", True),
            timeout=timedelta(minutes=float(cfg.get("timeout_minutes", 10))),
        )
    else:
        rank_zero_info(
            "INFO: Option 0: pl_strategy = FSDPStrategy(sharding_strategy=...)"
        )
        strategy_kwargs["sharding_strategy"] = SHARDING_STRATEGY.get(
            cfg.sharding_strategy
        )

    if "awp" in strategy_str:
        rank_zero_info("INFO: FSDP - Auto-Wrap-Policy")
        auto_wrap_policy = {TensorProductConvLayer}
        strategy_kwargs["auto_wrap_policy"] = auto_wrap_policy

    if "ac" in strategy_str:
        rank_zero_info("INFO: FSDP - Activation Checkpointing")
        ac_policy = {TensorProductConvLayer}
        strategy_kwargs["activation_checkpointing_policy"] = ac_policy

    precision = cfg.get("precision", None)
    if precision is not None:
        rank_zero_info(f"Precision={precision}")
        precision_plugin = FSDPPrecision(precision=precision)
        strategy_kwargs["precision_plugin"] = precision_plugin

    pl_strategy = FSDPStrategy(**strategy_kwargs)

    return pl_strategy


def setup_logger(cfg) -> Optional[Logger]:
    logger = None
    logger_cfg = cfg.logger
    if logger_cfg.wandb:
        logger_cfg = logger_cfg
        logger = WandbLogger(
            entity=logger_cfg.entity,
            project=logger_cfg.project,
            name=logger_cfg.name,
            tags=logger_cfg.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    return logger


def main(config_file, args):
    torch.set_float32_matmul_precision("high")
    # SIGUSR2 is a non-destructive on-demand traceback for all threads.  It is
    # invaluable for diagnosing a rank that has stopped before a DDP
    # collective; normal NCCL timeout output only identifies the waiting
    # ranks, not the stalled Python frame.
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR2"):
        faulthandler.register(signal.SIGUSR2, all_threads=True)

    assert os.path.exists(config_file)
    raw_config = OmegaConf.load(config_file)

    # Apply input arguments
    args = OmegaConf.from_dotlist(args)
    cfg = OmegaConf.merge(raw_config, args)
    OmegaConf.resolve(cfg)

    logging.getLogger().setLevel("INFO")

    rank_zero_info(f"Running with seed {cfg.seed}")
    seed_everything(cfg.seed)

    data_module = setup_training_datamodule(
        data_cfg=cfg.data, transform_cfg=cfg.transforms
    )

    run_dir = os.path.join(cfg.log_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    model = setup_model(cfg, task=cfg.data.task)
    # Weight-only restarts intentionally reset Lightning's local epoch.  Keep
    # the externally reported epoch monotonic so loss rows, plots and
    # PoseBusters snapshots are never overwritten or mislabeled.
    history_epoch_offset = 0
    history_path = os.path.join(run_dir, "loss_history.json")
    try:
        history_rows = json.load(open(history_path))
        history_epoch_offset = max(
            (int(row.get("epoch", 0)) for row in history_rows if isinstance(row, dict)),
            default=0,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    model.history_epoch_offset = 0
    initial_checkpoint = cfg.get("init_model_ckpt", None)
    checkpoint_epoch = None
    if initial_checkpoint:
        if not os.path.isfile(initial_checkpoint):
            raise FileNotFoundError(f"init_model_ckpt does not exist: {initial_checkpoint}")
        checkpoint_epoch = load_initial_model_weights(model, initial_checkpoint)

    strategy = setup_strategy(cfg.strategy)
    callbacks = setup_callbacks(args=cfg.callbacks, run_dir=run_dir)
    logger = setup_logger(cfg=cfg)

    trainer_cfg = cfg.trainer
    trainer = pl.Trainer(
        **trainer_cfg, strategy=strategy, callbacks=callbacks, logger=logger
    )
    if (
        initial_checkpoint
        and not cfg.get("restart_ckpt", None)
        and cfg.get("preserve_init_checkpoint_epoch", True)
    ):
        set_weight_only_start_epoch(
            trainer,
            checkpoint_epoch
            if checkpoint_epoch is not None
            else history_epoch_offset,
        )

    config_out = os.path.join(run_dir, "model_parameters.yml")
    save_config(cfg, config_out)

    # fit model
    trainer.fit(
        model=model, datamodule=data_module, ckpt_path=cfg.get("restart_ckpt", None)
    )


if __name__ == "__main__":
    config_file = sys.argv[1]
    args = sys.argv[2:]

    main(config_file, args)
