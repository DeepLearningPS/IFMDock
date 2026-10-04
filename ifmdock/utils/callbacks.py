import json
import math
import os
import time
import subprocess
import shutil
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import rank_zero_only


class LossCurveCallback(Callback):
    """Persist every epoch's losses and periodically refresh the curve plot."""

    def __init__(self, run_dir, every_n_epochs: int = 10):
        self.run_dir = Path(run_dir)
        self.every_n_epochs = every_n_epochs
        self.history_path = self.run_dir / "loss_history.json"

    def _load_history(self):
        """Read prior history so resumed runs extend, rather than replace, it."""
        if not self.history_path.is_file():
            return []
        try:
            history = json.loads(self.history_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Keep training recoverable even if a previous process was killed
            # while writing the history.  The next successful epoch rewrites it.
            return []
        return history if isinstance(history, list) else []

    def _write_history(self, history):
        """Atomically replace history so an interrupted write cannot corrupt it."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.history_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(history, indent=2) + "\n")
        tmp_path.replace(self.history_path)

    def _plot_history(self, history):
        """Render the accumulated history without changing its on-disk record."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Derive the panels from persisted rows, so a later model version can
        # add a loss component without requiring a callback rewrite.
        components = sorted(
            {
                key.removeprefix("train_")
                for item in history
                for key in item
                if key.startswith("train_")
                and (key == "train_loss" or key.endswith("_loss"))
            }
            | {
                key.removeprefix("val_")
                for item in history
                for key in item
                if key.startswith("val_")
                and (key == "val_loss" or key.endswith("_loss"))
            }
        )
        if not components:
            return
        ncols = 2
        nrows = math.ceil(len(components) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows), squeeze=False)
        for ax, component in zip(axes.flat, components):
            for stage in ("train", "val"):
                key = f"{stage}_{component}"
                points = [(item["epoch"], item[key]) for item in history if key in item]
                if points:
                    x, y = zip(*points)
                    ax.plot(x, y, marker="o", label=stage)
            ax.set(xlabel="epoch", ylabel="loss", title=component)
            ax.grid(alpha=0.25)
            ax.legend()
        for ax in axes.flat[len(components):]:
            ax.set_visible(False)
        fig.suptitle("Rigid-pocket docking losses")
        fig.tight_layout()
        fig.savefig(self.run_dir / "loss_curve.png", dpi=160)
        plt.close(fig)

    @staticmethod
    def _collect_loss_values(metrics, prefix):
        """Extract one stage's scalar losses from Lightning's metric store."""
        values = {}
        for key, value in metrics.items():
            if (
                key.startswith(prefix)
                and (key == f"{prefix}loss" or key.endswith("_loss"))
                and value is not None
            ):
                values[key] = float(value.detach().cpu())
        return values

    def _upsert_epoch(self, epoch, values):
        """Merge a train or validation stage into that epoch's durable row."""
        history = self._load_history()
        row = next((item for item in history if item.get("epoch") == epoch), None)
        if row is None:
            row = {"epoch": epoch}
            history.append(row)
        row.update(values)
        history.sort(key=lambda item: item["epoch"])
        self._write_history(history)
        return history

    def on_train_epoch_end(self, trainer, pl_module):
        # Lightning clears some train result metrics during validation.  Save
        # them before validation starts, then merge validation metrics below.
        #
        # Every DDP rank must read ``callback_metrics`` here.  Accessing an
        # epoch metric can finalize its ``sync_dist`` reduction; decorating
        # this hook with ``rank_zero_only`` therefore made rank zero enter an
        # all-reduce while the other ranks entered ModelCheckpoint's barrier.
        # Only the filesystem mutation is rank-zero-only.
        epoch = trainer.current_epoch + 1 + getattr(pl_module, "history_epoch_offset", 0)
        values = self._collect_loss_values(trainer.callback_metrics, "train_")
        if trainer.is_global_zero and values:
            history = self._upsert_epoch(epoch, values)
            # Lightning runs validation and its callback before this hook.
            # Plot here, after both stages have been merged into the row, so
            # epoch 10/20/... never has one fewer train point than val points.
            if epoch % self.every_n_epochs == 0:
                self._plot_history(history)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1 + getattr(pl_module, "history_epoch_offset", 0)
        values = self._collect_loss_values(trainer.callback_metrics, "val_")
        if not trainer.is_global_zero or not values:
            return
        history = self._upsert_epoch(epoch, values)



class ModelWeightsCallback(Callback):
    """Save one compact inference checkpoint containing only EMA parameters.

    Lightning's resumable checkpoint also contains Adam moments and the EMA
    shadow copy, which is necessary for exact continuation but roughly five
    times larger than the network weights.  This companion file is intended
    for inference, sampling and archiving; ``last_model.pt`` remains the sole
    resumable checkpoint.
    """

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    @staticmethod
    def _payload(trainer, pl_module, epoch):
        state_dict = {
            name: value.detach().cpu().clone()
            for name, value in pl_module.state_dict().items()
        }
        if getattr(pl_module, "ema", None) is not None:
            trainable_names = [
                f"model.{name}"
                for name, parameter in pl_module.model.named_parameters()
                if parameter.requires_grad
            ]
            for name, value in zip(trainable_names, pl_module.ema.shadow_params):
                state_dict[name] = value.detach().cpu().clone()
        return {
            "format": "ifmdock_weights_only_v1",
            "epoch": epoch,
            "global_step": trainer.global_step,
            "state_dict": state_dict,
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1 + getattr(pl_module, "history_epoch_offset", 0)
        if trainer.is_global_zero:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            payload = self._payload(trainer, pl_module, epoch)
            target = self.run_dir / "model_weights.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, target)
        # All ranks must leave this callback together.  Otherwise non-zero
        # ranks can enter the next backward pass while rank zero is still
        # serializing roughly 0.5 GB of weights.
        trainer.strategy.barrier("model_weights_saved")


class PeriodicTrainWeightsCallback(Callback):
    """Save compact EMA inference weights at train epochs without validation."""

    def __init__(self, run_dir, every_n_epochs: int):
        self.run_dir = Path(run_dir)
        self.every_n_epochs = int(every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1 + getattr(pl_module, "history_epoch_offset", 0)
        if epoch % self.every_n_epochs == 0 and trainer.is_global_zero:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            payload = ModelWeightsCallback._payload(trainer, pl_module, epoch)
            target = self.run_dir / f"epoch_{epoch:04d}_weights.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, target)
            latest = self.run_dir / "model_weights.pt"
            latest_tmp = latest.with_suffix(".pt.tmp")
            shutil.copy2(target, latest_tmp)
            os.replace(latest_tmp, latest)
        trainer.strategy.barrier("periodic_train_weights_saved")


class TrainTorsionEarlyStopping(Callback):
    """Stop when the epoch-level training torsion loss no longer improves.

    This is deliberately independent of validation because some fine-data
    runs use every available complex for training.  On an exact Lightning
    restart its state is restored from the checkpoint; for older checkpoints
    that predate this callback, the persisted loss history initializes the
    best value and stale-epoch count.
    """

    def __init__(self, run_dir, patience: int = 50, min_delta: float = 0.0):
        self.run_dir = Path(run_dir)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_epoch = 0
        self.stale = 0
        self._restored = False

    @property
    def state_key(self):
        return f"{type(self).__qualname__}[patience={self.patience},min_delta={self.min_delta}]"

    def state_dict(self):
        return {"best": self.best, "best_epoch": self.best_epoch, "stale": self.stale}

    def load_state_dict(self, state_dict):
        self.best = float(state_dict.get("best", float("inf")))
        self.best_epoch = int(state_dict.get("best_epoch", 0))
        self.stale = int(state_dict.get("stale", 0))
        self._restored = True

    def on_fit_start(self, trainer, pl_module):
        if self._restored:
            return
        history_path = self.run_dir / "loss_history.json"
        try:
            rows = json.loads(history_path.read_text())
            points = [
                (int(row["epoch"]), float(row["train_tor_loss"]))
                for row in rows
                if isinstance(row, dict) and row.get("train_tor_loss") is not None
                and math.isfinite(float(row["train_tor_loss"]))
            ]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            points = []
        if points:
            self.best_epoch, self.best = min(points, key=lambda item: item[1])
            self.stale = max(0, max(epoch for epoch, _ in points) - self.best_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get("train_tor_loss")
        if metric is None:
            raise RuntimeError("Torsion early stopping requires train_tor_loss")
        value = float(metric.detach().cpu())
        epoch = trainer.current_epoch + 1 + getattr(pl_module, "history_epoch_offset", 0)
        if math.isfinite(value) and value < self.best - self.min_delta:
            self.best, self.best_epoch, self.stale = value, epoch, 0
        else:
            self.stale += 1
        if trainer.is_global_zero:
            target = self.run_dir / "torsion_early_stopping_state.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "epoch": epoch, "train_tor_loss": value,
                "best_train_tor_loss": self.best, "best_epoch": self.best_epoch,
                "stale_epochs": self.stale, "patience": self.patience,
                "min_delta": self.min_delta,
            }, indent=2) + "\n")
            os.replace(temporary, target)
        # Every rank observes the same sync_dist epoch metric and therefore
        # makes the same stop decision at the safe epoch boundary.
        if self.stale >= self.patience:
            trainer.should_stop = True
            if trainer.is_global_zero:
                print(
                    f"Torsion early stopping at epoch {epoch}: no improvement "
                    f"for {self.stale} epochs (best={self.best:.6g} at {self.best_epoch})",
                    flush=True,
                )


class EpochBoundaryEvaluationPause(Callback):
    """Pause all DDP ranks after a requested, fully persisted epoch."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Keep this directory in sync with the evaluation scripts. A previous
        # name (``posebusters_evaluations``)
        # meant the trainer never observed requests written by the monitor.
        request = self.run_dir / "posebusters_convergence" / "evaluation_request.json"
        if not request.is_file():
            return
        try:
            requested_epoch = int(json.loads(request.read_text())["epoch"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return
        epoch = trainer.current_epoch + 1 + getattr(pl_module, "history_epoch_offset", 0)
        if epoch < requested_epoch:
            return
        trainer.strategy.barrier("posebusters_epoch_complete")
        torch.cuda.empty_cache()
        if trainer.is_global_zero:
            ready = request.with_name("evaluation_ready.json")
            tmp = ready.with_suffix(".tmp")
            tmp.write_text(json.dumps({"epoch": epoch}) + "\n")
            os.replace(tmp, ready)
        trainer.strategy.barrier("posebusters_ready")
        while request.exists():
            time.sleep(2)
        trainer.strategy.barrier("posebusters_complete")


class ConvergedFiveMetricEvaluation(Callback):
    """Synchronous epoch-boundary evaluation without restarting training."""

    def __init__(self, run_dir, patience=50, interval=20, gpu_ids="0,1,2"):
        self.run_dir = Path(run_dir)
        self.patience, self.interval = int(patience), int(interval)
        self.gpu_ids = gpu_ids
        self.best_loss = float("inf")
        self.stale = 0
        self.converged_epoch = None
        self.last_evaluation = None
        self.best_rates = {"ifmscore_top1": -1.0, "average_rmsd": -1.0}
        for key in self.best_rates:
            path = self.run_dir / f"best_{key}_metrics.json"
            if path.is_file():
                self.best_rates[key] = float(json.loads(path.read_text())[key])

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            "best_loss", "stale", "converged_epoch", "last_evaluation", "best_rates"
        )}

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

    def on_train_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get("val_loss")
        if metric is None:
            raise RuntimeError("Convergence evaluation requires val_loss")
        value = float(metric.detach().cpu())
        epoch = trainer.current_epoch + 1
        if math.isfinite(value) and value < self.best_loss:
            self.best_loss, self.stale = value, 0
        else:
            self.stale += 1
        if self.converged_epoch is None and self.stale >= self.patience:
            self.converged_epoch = epoch
        if trainer.is_global_zero:
            (self.run_dir / "convergence_state.json").write_text(
                json.dumps({"epoch": epoch, "val_loss": value, **self.state_dict()}, indent=2) + "\n")
        due = self.converged_epoch is not None and (
            (self.last_evaluation is None and epoch >= self.converged_epoch + self.interval)
            or (self.last_evaluation is not None and epoch >= self.last_evaluation + self.interval)
        )
        if not due:
            return
        trainer.strategy.barrier("five_metric_epoch_boundary")
        torch.cuda.empty_cache()
        status_path = self.run_dir / f"evaluation_status_{epoch:04d}.json"
        if trainer.is_global_zero:
            status_path.unlink(missing_ok=True)
        trainer.strategy.barrier("five_metric_status_reset")
        outcome = {"ok": True}
        if trainer.is_global_zero:
            try:
                root = Path(__file__).resolve().parents[2]
                log_dir = self.run_dir / "online_evaluation_logs"
                log_dir.mkdir(exist_ok=True)
                with (log_dir / f"epoch_{epoch:04d}.log").open("w") as log:
                    child_env = dict(os.environ)
                    for key in list(child_env):
                        if key in {
                            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                            "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                            "ROLE_WORLD_SIZE", "NODE_RANK",
                        } or key.startswith("TORCHELASTIC_"):
                            child_env.pop(key, None)
                    subprocess.run([
                        "bash", str(root / "scripts/evaluation/evaluate_stage1_five_metrics_gpu_group.sh"),
                        self.run_dir.name, str(epoch), self.gpu_ids,
                    ], cwd=root, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
                summary = json.loads((self.run_dir / "posebusters_five_metrics" /
                    f"epoch_{epoch:04d}" / "full_evaluation_summary.json").read_text())
                rates = summary["five_metrics_over_all_requested"]
                with (self.run_dir / "online_five_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch, **rates}) + "\n")
                for key in self.best_rates:
                    if rates[key] > self.best_rates[key]:
                        self.best_rates[key] = rates[key]
                        shutil.copy2(self.run_dir / "model_weights.pt", self.run_dir / f"best_{key}_model.pt")
                        (self.run_dir / f"best_{key}_metrics.json").write_text(
                            json.dumps({"epoch": epoch, **rates}, indent=2) + "\n")
                print(f"Epoch {epoch} five metrics: {rates}", flush=True)
            except Exception as error:
                outcome = {"ok": False, "error": str(error)}
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(outcome) + "\n")
            os.replace(temporary, status_path)
        else:
            # Do not enqueue an NCCL broadcast while rank zero evaluates:
            # long physical checks can exceed the process-group timeout.
            while not status_path.is_file():
                time.sleep(2)
        outcome = trainer.strategy.broadcast(outcome, src=0)
        if not outcome["ok"]:
            raise RuntimeError(f"Online evaluation failed: {outcome['error']}")
        self.last_evaluation = epoch
        self.best_rates = trainer.strategy.broadcast(self.best_rates, src=0)
        trainer.strategy.barrier("five_metric_training_resume")


def setup_docking_callbacks(args, run_dir):
    # A separate full ``best`` checkpoint duplicates optimizer and EMA state.
    # ``model_weights.pt`` below is the lightweight artifact for inference.
    callbacks = [ModelWeightsCallback(run_dir)]

    periodic_train_weights = getattr(args, "periodic_train_weights_every_n_epochs", 0)
    if periodic_train_weights:
        callbacks.append(PeriodicTrainWeightsCallback(run_dir, periodic_train_weights))

    if getattr(args, "save_last_model", True):
        last_model_checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="last_model",
            monitor=None,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_top_k=1,
            enable_version_counter=False,
        )
        last_model_checkpoint.FILE_EXTENSION = ".pt"
        callbacks.append(last_model_checkpoint)

    if getattr(args, "torsion_early_stopping", False):
        callbacks.append(TrainTorsionEarlyStopping(
            run_dir,
            patience=getattr(args, "torsion_early_stopping_patience", 50),
            min_delta=getattr(args, "torsion_early_stopping_min_delta", 0.0),
        ))

    loss_plot_every_n_epochs = getattr(args, "loss_plot_every_n_epochs", 10)
    if loss_plot_every_n_epochs:
        callbacks.append(
            LossCurveCallback(
                run_dir=run_dir, every_n_epochs=loss_plot_every_n_epochs
            )
        )
    callbacks.append(EpochBoundaryEvaluationPause(run_dir))
    if getattr(args, "online_five_metric_evaluation", False):
        callbacks.append(ConvergedFiveMetricEvaluation(
            run_dir, patience=getattr(args, "convergence_patience", 50),
            interval=getattr(args, "evaluation_interval", 20),
            gpu_ids=getattr(args, "evaluation_gpu_ids", "0,1,2"),
        ))

    if args.val_inference_freq is not None:
        for metric in args.inference_earlystop_metric.split(","):
            best_inf_checkpoint = ModelCheckpoint(
                dirpath=run_dir,
                filename=f"best_inference_epoch_model_{metric}",
                monitor=metric,
                mode=args.inference_earlystop_goal,
                every_n_epochs=args.val_inference_freq,
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
            best_inf_checkpoint.FILE_EXTENSION = ".pt"
            callbacks.append(best_inf_checkpoint)

        if args.flexible_sidechains:
            best_sc_checkpoint = ModelCheckpoint(
                dirpath=run_dir,
                filename="best_inference_epoch_model_aa",
                monitor="valinf_aa_rmsds_lt1",
                mode="max",
                every_n_epochs=args.val_inference_freq,
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
            best_sc_checkpoint.FILE_EXTENSION = ".pt"
            callbacks.append(best_sc_checkpoint)

        if args.flexible_backbone:
            best_bb_checkpoint = ModelCheckpoint(
                dirpath=run_dir,
                filename="best_inference_epoch_model_bb",
                monitor="valinf_bb_rmsds_lt1",
                mode="max",
                every_n_epochs=args.val_inference_freq,
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
            best_bb_checkpoint.FILE_EXTENSION = ".pt"
            callbacks.append(best_bb_checkpoint)

    if args.wandb:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    return callbacks


def setup_filtering_callbacks(args, run_dir):
    best_loss_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_loss",
        monitor="val_filtering_loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    best_loss_checkpoint.FILE_EXTENSION = ".pt"

    best_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        monitor="val_accuracy",
        mode="max",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    # By default this saves as ".ckpt"
    best_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks = [best_loss_checkpoint, best_model_checkpoint]

    if args.atom_lig_confidence:
        best_atom_loss_checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best_atom_loss",
            monitor="val_atom_filtering_loss",
            mode="min",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_top_k=1,
        )
        best_atom_loss_checkpoint.FILE_EXTENSION = ".pt"

        best_atom_model_checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best_atom_model",
            monitor="val_atom_accuracy",
            mode="max",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_top_k=1,
        )
        best_atom_model_checkpoint.FILE_EXTENSION = ".pt"
        callbacks.extend([best_atom_loss_checkpoint, best_atom_model_checkpoint])

    if args.wandb:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    return callbacks


def setup_relaxation_callbacks(args, run_dir):
    callbacks = []
    best_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        monitor=args.main_metric,
        mode=args.main_metric_goal,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    best_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks.append(best_model_checkpoint)

    last_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="last_model",
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    last_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks.append(last_model_checkpoint)

    if args.val_inference_freq is not None:
        best_inf_checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best_inference_epoch_model",
            monitor=args.inference_earlystop_metric,
            mode=args.inference_earlystop_goal,
            every_n_epochs=args.val_inference_freq,
            save_on_train_epoch_end=True,
            save_top_k=1,
        )
        best_inf_checkpoint.FILE_EXTENSION = ".pt"
        callbacks.append(best_inf_checkpoint)

    if args.wandb:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    return callbacks


def setup_callbacks(args, run_dir, task: str = "docking"):
    if task == "docking":
        return setup_docking_callbacks(args=args, run_dir=run_dir)

    elif task == "relaxation":
        return setup_relaxation_callbacks(args=args, run_dir=run_dir)

    else:
        assert task == "filtering"
        return setup_filtering_callbacks(args=args, run_dir=run_dir)
