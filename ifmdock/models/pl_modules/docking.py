import copy
from typing import Any
from functools import partial
import contextlib

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
import torch
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import to_dense_batch
import timeit
import logging
import os

from ifmdock.data.conformers.protein import scRMSD

# from ifmdock.data.transforms.docking.bb_priors import construct_bb_prior
from ifmdock.geometry.ops import rigid_transform_kabsch_numpy

from ifmdock.models.networks import get_model
from ifmdock.models.loss.docking import IFMDockLoss
from ifmdock.models.optim.lr_schedulers import AlphaFoldLRScheduler
from ifmdock.models.optim.ema import ExponentialMovingAverage
from ifmdock.metrics.docking import pli_lddt_score, CustomMeanMetric

from ifmdock.sampling.docking.diffusion import t_to_sigma as t_to_sigma_compl
from ifmdock.sampling.docking import sampling, sampling_fast


class IFMDockModule(pl.LightningModule):
    def __init__(
        self, model_cfg, sigma_cfg, training_cfg, sampler_cfg, loss_cfg=None,
        physical_loss_cfg=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.sigma_cfg = sigma_cfg
        self.sampler_cfg = sampler_cfg
        self.training_cfg = training_cfg
        self.physical_loss_cfg = physical_loss_cfg

        self.t_to_sigma = t_to_sigma = partial(t_to_sigma_compl, args=sigma_cfg)
        self.model = get_model(
            args=model_cfg,
            t_to_sigma=t_to_sigma,
            confidence_mode=False,
            device=self.device,  # PyL doesn't need a device argument
        )
        if loss_cfg is not None:
            self.loss = IFMDockLoss(args=loss_cfg, t_to_sigma=t_to_sigma)
        # Move this into sampler
        # self.bb_prior = construct_bb_prior(args)
        self.bb_prior = None
        self.ema = None
        self.setup_metrics()

    def setup_metrics(self):
        """Setup torchmetrics"""
        self.metrics_dict = torch.nn.ModuleDict()
        for metric in [
            "rmsds_lt1",
            "rmsds_lt2",
            "rmsds_lt5",
            "bb_rmsds_lt1",
            "bb_rmsds_lt2",
            "bb_rmsds_lt05",
            "aa_rmsds_lt1",
            "aa_rmsds_lt2",
            "aa_rmsds_lt05",
            "pli_lddt",
        ]:
            # nan vals possible for certain metrics
            self.metrics_dict[metric] = CustomMeanMetric()

    def training_step(self, batch, batch_idx):
        self._trace_batch_idx = batch_idx
        if os.environ.get("IFMDOCK_TRACE_BATCHES"):
            trace_dir = os.environ["IFMDOCK_TRACE_BATCHES"]
            os.makedirs(trace_dir, exist_ok=True)
            names = getattr(batch, "name", [])
            lig_nodes = int(batch["ligand"].num_nodes)
            atom_nodes = int(batch["atom"].num_nodes)
            lig_edges = int(batch["ligand", "lig_bond", "ligand"].edge_index.shape[1])
            with open(os.path.join(trace_dir, f"rank_{self.global_rank}.log"), "a") as handle:
                handle.write(
                    f"start epoch={self.current_epoch + 1} batch={batch_idx} "
                    f"lig_nodes={lig_nodes} atom_nodes={atom_nodes} "
                    f"lig_edges={lig_edges} names={list(names)}\n"
                )
                handle.flush()
        predictions = self.general_step_with_oom(batch, batch_idx)
        if os.environ.get("IFMDOCK_TRACE_BATCHES"):
            with open(os.path.join(os.environ["IFMDOCK_TRACE_BATCHES"], f"rank_{self.global_rank}.log"), "a") as handle:
                handle.write(f"forward_done epoch={self.current_epoch + 1} batch={batch_idx}\n")
                handle.flush()
        if getattr(self.model, "cartesian_refinement", False):
            loss, loss_breakdown = self._cartesian_flow_loss(predictions, batch)
        else:
            loss, loss_breakdown = self.loss(predictions, batch, apply_mean=True)
        loss, loss_breakdown = self._add_distance_guidance_loss(
            loss, loss_breakdown, batch
        )

        self._log_loss_breakdown("train", loss_breakdown)
        if os.environ.get("IFMDOCK_TRACE_BATCHES"):
            with open(os.path.join(os.environ["IFMDOCK_TRACE_BATCHES"], f"rank_{self.global_rank}.log"), "a") as handle:
                handle.write(f"step_done epoch={self.current_epoch + 1} batch={batch_idx}\n")
                handle.flush()
        return loss

    def on_after_backward(self):
        if os.environ.get("IFMDOCK_TRACE_BATCHES"):
            with open(os.path.join(os.environ["IFMDOCK_TRACE_BATCHES"], f"rank_{self.global_rank}.log"), "a") as handle:
                handle.write(
                    f"backward_done epoch={self.current_epoch + 1} "
                    f"batch={getattr(self, '_trace_batch_idx', -1)}\n"
                )
                handle.flush()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        self.stage = "val"

        # Validation dataloader
        if dataloader_idx == 0:
            with torch.no_grad():
                predictions = self.general_step_with_oom(batch, batch_idx)

            if getattr(self.model, "cartesian_refinement", False):
                loss, loss_breakdown = self._cartesian_flow_loss(predictions, batch)
            else:
                loss, loss_breakdown = self.loss(predictions, batch, apply_mean=True)
            _, loss_breakdown = self._add_distance_guidance_loss(
                loss, loss_breakdown, batch
            )
            self._log_loss_breakdown("val", loss_breakdown)

        # Inference data loader
        elif dataloader_idx == 1:
            val_inference_freq = self.training_cfg.val_inference_freq
            if (
                val_inference_freq is not None
                and (self.trainer.current_epoch + 1) % val_inference_freq == 0
            ):
                self.stage = "valinf"
                # Need to specify to use apo positions if not using flexible sidechain or flexible backbone.
                # During training this is done via ProteinTransform but not applied during inference
                if (
                    not self.model_cfg.flexible_backbone
                    and not self.model_cfg.flexible_sidechains
                ):
                    batch["atom"].pos = batch["atom"].orig_aligned_apo_pos
                    batch["receptor"].pos = batch["atom"].pos[batch["atom"].ca_mask]

                # This is needed for bf16-mixed, since torch.linalg.svd is not supported in bf16
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    inf_predictions = self.run_inference(batch, batch_idx)
                    self._compute_inference_metrics(
                        batch=batch, predictions=inf_predictions, batch_idx=batch_idx
                    )

    def _log_loss_breakdown(self, stage, loss_breakdown):
        """Log a fixed metric set in the same collective order on every rank.

        A mini-batch can contain no rotatable ligand bond on one DDP rank but
        not another.  Logging a rank-dependent set of torsion metrics makes
        Lightning's epoch metric reduction enter collectives in different
        orders.  Keeping the full rigid-ligand loss breakdown fixed, sanitising the
        diagnostic torsion value, and using ``sync_dist`` gives a true global
        mean without that mismatch.
        """
        for key in (
            "loss",
            "diffusion_loss",
            "flow_loss",
            "tr_base_loss",
            "rot_base_loss",
            "tor_base_loss",
            "tr_loss",
            "rot_loss",
            "tor_loss",
            "distance_loss",
            "cross_distance_loss",
            "holo_distance_loss",
            "cartesian_velocity_loss",
            "bond_geometry_loss",
            "angle_geometry_loss",
            "ligand_steric_loss",
            "protein_ligand_overlap_loss",
            "protein_ligand_overlap_raw",
            "physical_time_weight",
            "physical_loss",
        ):
            value = torch.nan_to_num(
                loss_breakdown.get(key, loss_breakdown["loss"] * 0.0),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            self.log(
                f"{stage}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
                add_dataloader_idx=False if stage == "val" else True,
            )

    def _cartesian_flow_loss(self, predictions, batch):
        pred = predictions["ligand_velocity"]
        target = batch["ligand"].cartesian_velocity_target
        atom_sq = torch.sum((pred - target) ** 2, dim=-1)
        graph_loss = scatter_mean(atom_sq, batch["ligand"].batch, dim=0)
        flow_loss = graph_loss.mean()
        physical, physical_terms = self._cartesian_physical_loss(pred, batch)
        loss = flow_loss + physical
        zero = loss.detach().new_zeros(())
        keys = (
            "diffusion_loss", "tr_base_loss", "rot_base_loss", "tor_base_loss",
            "tr_loss", "rot_loss", "tor_loss", "distance_loss",
            "cross_distance_loss", "holo_distance_loss",
        )
        out = {key: zero for key in keys}
        out.update({
            "loss": loss.detach(), "flow_loss": flow_loss.detach(),
            "cartesian_velocity_loss": flow_loss.detach(),
            "physical_loss": physical.detach(),
        })
        out.update({key: value.detach() for key, value in physical_terms.items()})
        return loss, out

    @staticmethod
    def _masked_graph_sum(
        values, graph_index, mask, num_graphs, graph_weights=None
    ):
        """Match IFMDock relaxation: sum violations per graph, then batch mean."""
        if not bool(mask.any()):
            return values.new_zeros(())
        per_graph = scatter_sum(
            values[mask], graph_index[mask], dim=0, dim_size=num_graphs
        )
        if graph_weights is not None:
            per_graph = per_graph * graph_weights
        return per_graph.mean()

    @staticmethod
    def _physical_time_weights(cfg, graph_t):
        """Return one physical-loss multiplier per graph, applied exactly once."""
        mode = str(cfg.get("time_weighting", "none")).lower()
        if mode in ("none", "constant"):
            return torch.ones_like(graph_t)
        if mode != "exponential":
            raise ValueError(f"Unknown physical time_weighting={mode!r}")

        alpha = float(cfg.get("time_weight_alpha", 3.0))
        floor = float(cfg.get("time_weight_floor", 0.0))
        if not 0.0 <= floor <= 1.0:
            raise ValueError("physical time_weight_floor must be in [0, 1]")
        t = graph_t.clamp(0.0, 1.0)
        if abs(alpha) < 1e-8:
            ramp = t
        else:
            ramp = torch.expm1(alpha * t) / torch.expm1(
                t.new_tensor(alpha)
            )
        return floor + (1.0 - floor) * ramp

    def _cartesian_physical_loss(self, velocity, batch):
        """IFMDock relaxation constraints evaluated on the predicted endpoint.

        The interpolated state itself is not required to be physical.  For
        straight flow matching, x_1_hat=x_t+(1-t)v is the appropriate object
        on which to impose molecular geometry and protein-overlap constraints.
        """
        cfg = self.physical_loss_cfg
        zero = velocity.new_zeros(())
        names = (
            "bond_geometry_loss", "angle_geometry_loss",
            "ligand_steric_loss", "protein_ligand_overlap_loss",
            "protein_ligand_overlap_raw",
            "physical_time_weight",
        )
        if cfg is None or not cfg.get("enabled", False):
            return zero, {name: zero for name in names}

        lig_batch = batch["ligand"].batch
        num_graphs = int(batch.num_graphs)
        graph_t = batch.complex_t["t"].to(velocity).reshape(-1)
        time_weights = self._physical_time_weights(cfg, graph_t)
        remaining = 1.0 - batch.complex_t["t"][lig_batch].unsqueeze(-1)
        endpoint = batch["ligand"].pos + remaining * velocity

        store = batch["ligand", "lig_edge", "ligand"]
        pairs = store.posebusters_edge_index
        pair_graph = lig_batch[pairs[0]]
        distances = torch.linalg.vector_norm(
            endpoint[pairs[1]] - endpoint[pairs[0]], dim=-1
        )
        lower = store.lower_bound.to(distances)
        upper = store.upper_bound.to(distances)
        bond_mask = store.posebusters_bond_mask.bool()
        angle_mask = store.posebusters_angle_mask.bool()
        geometry_buffer = float(cfg.get("geometry_buffer", 0.05))
        geometry_violation = (
            torch.relu(lower * (1.0 - geometry_buffer) - distances)
            + torch.relu(distances - upper * (1.0 + geometry_buffer))
        )
        bond_loss = self._masked_graph_sum(
            geometry_violation, pair_graph, bond_mask, num_graphs, time_weights
        )
        angle_loss = self._masked_graph_sum(
            geometry_violation, pair_graph, angle_mask, num_graphs, time_weights
        )

        nonlocal_mask = ~(bond_mask | angle_mask)
        steric_buffer = float(cfg.get("steric_buffer", 0.05))
        steric_violation = torch.relu(
            lower * (1.0 - steric_buffer) - distances
        )
        steric_loss = self._masked_graph_sum(
            steric_violation, pair_graph, nonlocal_mask, num_graphs, time_weights
        )

        lig_dense, lig_mask = to_dense_batch(endpoint, lig_batch)
        atom_dense, atom_mask = to_dense_batch(batch["atom"].pos, batch["atom"].batch)
        lig_radii, _ = to_dense_batch(batch["ligand"].vdw_radii, lig_batch)
        atom_radii, _ = to_dense_batch(
            batch["atom"].vdw_radii, batch["atom"].batch
        )
        cross_dist = torch.linalg.vector_norm(
            lig_dense.unsqueeze(2) - atom_dense.unsqueeze(1), dim=-1
        )
        allowed_overlap = float(cfg.get("overlap_buffer", 0.4))
        cross_violation = torch.relu(
            lig_radii.unsqueeze(2) + atom_radii.unsqueeze(1)
            - allowed_overlap - cross_dist
        )
        valid_pairs = lig_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
        cross_violation = cross_violation * valid_pairs
        # Original IFMDock relaxation sums all excessive VDW overlaps within
        # each complex and then averages complexes.  Averaging over every
        # ligand-protein pair would dilute the sparse clash signal to ~1e-5.
        overlap_per_graph = cross_violation.sum(dim=(1, 2))
        overlap_raw = overlap_per_graph.mean()
        reduction = str(cfg.get("overlap_reduction", "linear"))
        if reduction == "log1p":
            overlap_loss = (time_weights * torch.log1p(overlap_per_graph)).mean()
        elif reduction == "linear":
            overlap_loss = (time_weights * overlap_per_graph).mean()
        else:
            raise ValueError(f"Unknown overlap_reduction={reduction!r}")

        terms = {
            "bond_geometry_loss": bond_loss,
            "angle_geometry_loss": angle_loss,
            "ligand_steric_loss": steric_loss,
            "protein_ligand_overlap_loss": overlap_loss,
            "protein_ligand_overlap_raw": overlap_raw,
            "physical_time_weight": time_weights.mean(),
        }
        total = (
            float(cfg.get("bond_weight", 1.0)) * bond_loss
            + float(cfg.get("angle_weight", 1.0)) * angle_loss
            + float(cfg.get("steric_weight", 1.0)) * steric_loss
            + float(cfg.get("overlap_weight", 1.0)) * overlap_loss
        )
        return total, terms

    def _add_distance_guidance_loss(self, diffusion_loss, loss_breakdown, batch):
        """Add EC-Dock distance supervision when stage-2 guidance is enabled."""
        guidance = getattr(self.model, "distance_guidance", None)
        distance_loss = getattr(batch, "distance_guidance_loss", None)
        if guidance is None or distance_loss is None:
            distance_loss = diffusion_loss.new_zeros(())
            cross_distance_loss = diffusion_loss.new_zeros(())
            holo_distance_loss = diffusion_loss.new_zeros(())
            total_loss = diffusion_loss
        else:
            distance_loss = torch.nan_to_num(distance_loss)
            cross_distance_loss = torch.nan_to_num(
                getattr(batch, "distance_guidance_cross_loss", distance_loss * 0.0)
            )
            holo_distance_loss = torch.nan_to_num(
                getattr(batch, "distance_guidance_holo_loss", distance_loss * 0.0)
            )
            # A frozen distance predictor supplies discrete graph topology but
            # has no differentiable path to IFMDock through ``edge_index``.
            # Keep its losses as diagnostics without contaminating the
            # optimization loss, loss curves, convergence monitor or
            # comparisons with stage 1.  When distance parameters are enabled,
            # retain the intended joint objective.
            if guidance.update_parameters:
                total_loss = diffusion_loss + guidance.loss_weight * distance_loss
            else:
                total_loss = diffusion_loss
        is_flow = getattr(self.loss.args, "lig_transform_type", "diffusion") == "flow"
        zero = diffusion_loss.detach().new_zeros(())
        # Cartesian flow has already separated its base velocity loss from the
        # physical regularizers.  Do not overwrite that diagnostic with the
        # combined objective here.
        if not getattr(self.model, "cartesian_refinement", False):
            loss_breakdown["diffusion_loss"] = zero if is_flow else diffusion_loss.detach().clone()
            loss_breakdown["flow_loss"] = diffusion_loss.detach().clone() if is_flow else zero
        loss_breakdown["distance_loss"] = distance_loss.detach().clone()
        loss_breakdown["cross_distance_loss"] = cross_distance_loss.detach().clone()
        loss_breakdown["holo_distance_loss"] = holo_distance_loss.detach().clone()
        loss_breakdown["loss"] = total_loss.detach().clone()
        return total_loss, loss_breakdown

    def general_step_with_oom(self, batch, batch_idx):
        # Never swallow a rank-local forward failure under DDP.  Returning
        # ``None`` on one rank lets its peers enter gradient collectives and
        # turns the real CUDA/torch-cluster exception into a five-minute NCCL
        # timeout.  torchrun must see the original exception and stop all
        # ranks together.
        try:
            return self.model(batch, fast_updates=True)

        except RuntimeError as e:
            if getattr(self.trainer, "world_size", 1) > 1:
                raise
            if "out of memory" in str(e):
                logging.error("| WARNING: ran out of memory, skipping batch")
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

            elif "Input mismatch" in str(e):
                logging.error("| WARNING: weird torch_cluster error, skipping batch")
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    @torch.no_grad()
    def run_inference(self, batch, batch_idx):
        schedules = sampling.get_schedules(
            inference_steps=self.sampler_cfg.inference_steps,
            bb_tr_bridge_alpha=self.sampler_cfg.bb_tr_bridge_alpha,
            bb_rot_bridge_alpha=self.sampler_cfg.bb_rot_bridge_alpha,
            sidechain_tor_bridge=self.sampler_cfg.sidechain_tor_bridge,
            sc_tor_bridge_alpha=self.sampler_cfg.sc_tor_bridge_alpha,
            inf_sched_alpha=1,
            inf_sched_beta=1,
            sigma_schedule="expbeta",
            lig_transform_type=getattr(self.sampler_cfg, "lig_transform_type", "diffusion"),
        )
        data_list = [copy.deepcopy(batch.to("cpu"))]

        randomize_fn = sampling_fast.randomize_position_inf
        sampling_fn = sampling_fast.sampling

        randomize_fn(
            data_list=data_list,
            no_torsion=self.sampler_cfg.no_torsion,
            no_random=False,
            tr_sigma_max=self.sigma_cfg.tr_sigma_max,
            flexible_sidechains=self.model_cfg.flexible_sidechains,
            flexible_backbone=self.model_cfg.flexible_backbone,
            sidechain_tor_bridge=self.model_cfg.sidechain_tor_bridge,
            use_bb_orientation_feats=self.model_cfg.use_bb_orientation_feats,
            prior=self.bb_prior,
            lig_transform_type=getattr(self.sampler_cfg, "lig_transform_type", "diffusion"),
            rot_sigma_max=getattr(self.sampler_cfg, "flow_rot_sigma", self.sigma_cfg.rot_sigma_max),
            tor_sigma_max=getattr(self.sampler_cfg, "flow_tor_sigma", self.sigma_cfg.tor_sigma_max),
            flow_box_padding=getattr(self.sampler_cfg, "flow_box_padding", 10.0),
            flow_source_mode=getattr(self.sampler_cfg, "flow_source_mode", "matched_endpoint_noise"),
            flow_rot_prior=getattr(self.sampler_cfg, "flow_rot_prior", "gaussian"),
            flow_tor_prior=getattr(self.sampler_cfg, "flow_tor_prior", "gaussian"),
            flow_tor_uniform_prob=getattr(self.sampler_cfg, "flow_tor_uniform_prob", 0.25),
            flow_tor_empirical_prob=getattr(self.sampler_cfg, "flow_tor_empirical_prob", 0.5),
        )

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list is None:
            try:
                predictions_list, confidences = sampling_fn(
                    data_list=data_list,
                    model=self.model,
                    inference_steps=self.sampler_cfg.inference_steps,
                    schedules=schedules,
                    sidechain_tor_bridge=self.model_cfg.sidechain_tor_bridge,
                    device=self.device,
                    t_to_sigma=self.t_to_sigma,
                    model_args=self.sampler_cfg,
                    no_final_step_noise=True,
                    debug_backbone=False,
                    debug_sidechain=False,
                    use_bb_orientation_feats=self.model_cfg.use_bb_orientation_feats,
                )
            except Exception as e:
                if "failed to converge" in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        logging.warning(
                            "| WARNING: SVD failed to converge 5 times - skipping the complex"
                        )
                        break
                    logging.warning(
                        "| WARNING: SVD failed to converge - trying again with a new sample"
                    )
                elif "out of bounds" in str(e):
                    failed_convergence_counter += 1
                    logging.warning(f"Failing: DEBUG | WARNING: {str(e)}")
                    if failed_convergence_counter > 5:
                        logging.info(
                            "DEBUG: `index out of bound` for more than 5 times - skipping the complex"
                        )
                        break
                else:
                    failed_convergence_counter += 1
                    logging.warning(f"Failing: DEBUG | WARNING: {str(e)}")
                    if failed_convergence_counter > 5:
                        logging.warning(
                            f"DEBUG: Unexpected error `{e}` for more than 5 times - skipping the complex"
                        )
                        break

            if failed_convergence_counter > 5:
                pass

        return predictions_list

    def _compute_inference_metrics(self, batch, predictions, batch_idx=None):
        if predictions is None:
            logging.info(
                f"Skipping inference metrics for batch_idx={batch_idx} since predictions=None"
            )
            return

        if self.model_cfg.no_torsion:
            orig_center = batch.original_center.cpu().numpy()
            centered_lig_pos = batch["ligand"].pos.cpu().numpy()
            batch["ligand"].orig_pos = centered_lig_pos + orig_center

        filterHs = torch.not_equal(predictions[0]["ligand"].x[:, 0], 0).cpu().numpy()

        if isinstance(batch["ligand"].orig_pos, list):
            batch["ligand"].orig_pos = batch["ligand"].orig_pos[0]
        if isinstance(batch["atom"].orig_holo_pos, list):
            batch["atom"].orig_holo_pos = batch["atom"].orig_holo_pos[0]

        orig_pos = batch["ligand"].orig_pos
        if isinstance(orig_pos, torch.Tensor):
            orig_pos = orig_pos.cpu().numpy()

        ligand_pos = []
        atom_pos = []
        ligand_pos_before = []
        orig_atom_pos = batch["atom"].orig_holo_pos.numpy()

        try:
            nearby_atoms = batch["atom"].nearby_atoms.cpu().numpy()
        except Exception:
            nearby_atoms = np.full(len(batch["atom"].pos), True)

        for complex_graph in predictions:
            atom_p = complex_graph["atom"].pos.cpu().numpy()
            try:
                R, t, _ = rigid_transform_kabsch_numpy(
                    orig_atom_pos[nearby_atoms], atom_p[nearby_atoms]
                )

                ligand_p = complex_graph["ligand"].pos.cpu().numpy()[filterHs]
                ligand_pos_before.append(ligand_p)

                atom_p = (R @ atom_p.T).T + t
                ligand_p = (R @ ligand_p.T).T + t
                atom_pos.append(atom_p)
                ligand_pos.append(ligand_p)

            except Exception as e:
                if "did not converge" in str(e):
                    logging.error(
                        "DEBUG | WARNING: numpy.linalg.LinAlgError: SVD did not converge"
                    )
                else:
                    raise e
        # to skip if no metric is computed
        if len(ligand_pos) == 0 or len(atom_pos) == 0:
            logging.debug("No ligand or proteina atoms found. No metric is computed!")
            return

        atom_pos = np.asarray(atom_pos)
        ligand_pos = np.asarray(ligand_pos)
        ligand_pos_before = np.asarray(ligand_pos_before)

        # TODO compute RMSD alignment of the overall position of atoms, ligand and receptors with the original complex

        orig_ligand_pos = np.expand_dims(batch["ligand"].orig_pos[filterHs], axis=0)

        orig_center = batch.original_center
        if isinstance(orig_center, list):
            orig_center = orig_center[0]

        if len(orig_center.shape) == 1:
            orig_center = orig_center[None, None, :]
        elif len(orig_center.shape) == 2:
            orig_center = orig_center[None]

        if isinstance(orig_center, torch.Tensor):
            orig_center = orig_center.cpu().numpy()

        rmsd = np.sqrt(
            ((ligand_pos + orig_center - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1)
        )
        self.metrics_dict["rmsds_lt1"]([100 * (rmsd < 1.0)])
        self.metrics_dict["rmsds_lt2"]([100 * (rmsd < 2.0)])
        self.metrics_dict["rmsds_lt5"]([100 * (rmsd < 5.0)])

        # We log this regardless
        calpha_mask = batch["atom"].ca_mask.numpy()
        calpha_pred_atoms = atom_pos[:, calpha_mask]
        calpha_holo_atoms = orig_atom_pos[None, calpha_mask]
        calpha_rmsd = np.sqrt(
            ((calpha_pred_atoms - calpha_holo_atoms) ** 2).sum(axis=2).mean(axis=1)
        )

        self.metrics_dict["bb_rmsds_lt2"]([100 * (calpha_rmsd < 2.0)])
        self.metrics_dict["bb_rmsds_lt1"]([100 * (calpha_rmsd < 1.0)])
        self.metrics_dict["bb_rmsds_lt05"]([100 * (calpha_rmsd < 0.5)])

        # We log this regardless
        aa_rmsd = scRMSD(nearby_atoms, atom_pos[0], orig_atom_pos)
        self.metrics_dict["aa_rmsds_lt2"]([100 * (aa_rmsd < 2.0)])
        self.metrics_dict["aa_rmsds_lt1"]([100 * (aa_rmsd < 1.0)])
        self.metrics_dict["aa_rmsds_lt05"]([100 * (aa_rmsd < 0.5)])

        # log pli-lddt score
        try:
            pli_lddt = pli_lddt_score(
                rec_coords_predicted=torch.from_numpy(atom_pos).float(),
                lig_coords_predicted=torch.from_numpy(ligand_pos).float(),
                rec_coords_true=torch.from_numpy(orig_atom_pos).float(),
                lig_coords_true=torch.from_numpy(orig_ligand_pos - orig_center).float(),
            )
        except ValueError as e:
            logging.error(
                f"Assigning 0 to pli_lddt metric for {batch['name']} due to error: {e}"
            )
            pli_lddt = torch.tensor([0.0])
        self.metrics_dict["pli_lddt"](100 * pli_lddt)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Updates EMA parameters after optimizer.step()
        self.ema.update(self.model.parameters())

    def on_validation_start(self):
        self.ema.store(self.model.parameters())
        if self.training_cfg.use_ema:
            rank_zero_info("Copying EMA parameters into model before validation")
            self.ema.copy_to(self.model.parameters())

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.training_cfg.use_ema:
            # Validation normally materializes ``ema_state_dict`` after
            # temporarily copying EMA weights into the model.  Train-only
            # fine-tuning intentionally disables validation, so construct the
            # same inference state directly from the EMA shadow parameters.
            if hasattr(self, "ema_state_dict"):
                ema_weights = copy.deepcopy(self.ema_state_dict)
            else:
                ema_weights = copy.deepcopy(self.model.state_dict())
                trainable_names = [
                    name for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                ]
                for name, value in zip(trainable_names, self.ema.shadow_params):
                    ema_weights[name] = value.detach().clone()
            checkpoint["ema_weights"] = ema_weights
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.training_cfg.use_ema and "ema" in checkpoint:
            rank_zero_info("Loading EMA from checkpoint")
            self.ema = ExponentialMovingAverage(
                parameters=self.model.parameters(), decay=self.training_cfg.ema_rate
            )
            self.ema.load_state_dict(state_dict=checkpoint["ema"], device=self.device)
            # Just a sanity check that numbers look sensible
            rank_zero_info(
                f"decay={self.ema.decay}, num_updates={self.ema.num_updates}"
            )

    def on_train_start(self):
        if self.ema is None:
            rank_zero_info("Initializing EMA")
            self.ema = ExponentialMovingAverage(
                parameters=self.model.parameters(), decay=self.training_cfg.ema_rate
            )
        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = timeit.default_timer()

        # Subsample complexes according to cluster ids if provided
        self.trainer.train_dataloader.dataset.subsample_clusters()

    def on_validation_epoch_start(self) -> None:
        self.validation_epoch_start_time = timeit.default_timer()
        # Subsample complexes according to cluster ids if provided
        self.trainer.val_dataloaders[0].dataset.subsample_clusters()

    def on_validation_epoch_end(self) -> None:
        # log metrics at epoch level
        for key, metric in self.metrics_dict.items():
            # if called
            if len(metric.values) > 0:
                self.log(f"valinf_{key}", metric.compute(), sync_dist=True)
                metric.reset()

        if self.training_cfg.use_ema:
            self.ema_state_dict = copy.deepcopy(self.model.state_dict())
            rank_zero_info("Restoring Model parameters...")
            self.ema.restore(self.model.parameters())

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        r"""Overrides the PyTorch Lightning backward step and adds the OOM check."""
        # A rank-local swallowed OOM leaves every other DDP rank blocked in
        # gradient all-reduce forever.  This applies to both stages whenever
        # distributed training is active: fail loudly so torchrun terminates
        # every rank and preserves the original CUDA error.
        if getattr(self.trainer, "world_size", 1) > 1:
            loss.backward(*args, **kwargs)
            return
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    f"| WARNING: ran OOM error, skipping batch. Exception: {str(e)}"
                )
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    def on_before_optimizer_step(self, optimizer):
        if self.training_cfg.check_unused_params:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    logging.info(f"gradients were None for {name}")

        if self.training_cfg.check_nan_grads:
            had_nan_grads = False
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    had_nan_grads = True
                    logging.info(f"gradients were nan for {name}")
            if had_nan_grads and self.training_cfg.except_on_nan_grads:
                raise Exception(
                    "There were nan gradients and except_on_nan_grads was set to True"
                )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self.training_cfg.skip_nan_grad_updates:
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    logging.info(
                        f"Gradients were nan for {name}, and skip_nan_grad_updates was enabled."
                        " Zeroing grad for this batch."
                    )
                    self.optimizer_zero_grad(epoch, batch_idx, optimizer)
                    break

        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        optimizer_cls = (
            torch.optim.AdamW
            if self.training_cfg.adamw == "adamw"
            else torch.optim.Adam
        )
        optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.training_cfg.lr),
            weight_decay=self.training_cfg.w_decay,
        )

        scheduler = None
        if self.training_cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.training_cfg.inference_earlystop_goal.split(",")[0]
                if self.training_cfg.val_inference_freq is not None
                else "min",
                factor=0.7,
                patience=self.training_cfg.scheduler_patience,
                min_lr=float(self.training_cfg.lr) / 100,
            )
        elif self.training_cfg.scheduler == "cosineannealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=250,  # {800, 500, 250, 100}
                eta_min=1e-7,  # default: 0
            )
        elif self.training_cfg.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.99  # first choice tried: 0.95
            )
        elif self.training_cfg.scheduler == "AlphaFoldLRScheduler":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                last_epoch=-1,
                warmup_no_steps=3,  # PLINDER: 1  ; PDBBind: 3
                start_decay_after_n_steps=125,  # PLINDER: 50  ;  PDBBind: 125
                decay_every_n_steps=2,  # PLINDER: 1  ; PDBBind: 2
                decay_factor=0.99,  # PLINDER: 0.98  ;  PDBBind: 0.99
            )
        else:
            rank_zero_info("No scheduler")
            scheduler = None

        optim_dict = {"optimizer": optimizer}

        if scheduler is not None:
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": self.training_cfg.inference_earlystop_metric.split(",")[0]
                if self.training_cfg.val_inference_freq is not None
                else "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
                "name": self.training_cfg.scheduler,
            }

        return optim_dict


def _make_model_backward_compatible():
    # Restructured codebase a bit, so need this hack to ensure things load correctly
    import sys
    from ifmdock.sampling.docking import diffusion

    sys.modules["ifmdock.utils.diffusion"] = diffusion
    sys.modules["ifmdock.components.docking.utils.diffusion"] = diffusion
    return


def load_pretrained_model(
    cfg, ckpt_file, use_ema_weights: bool = False, freeze: bool = False
):
    model = IFMDockModule(
        model_cfg=cfg.model,
        sigma_cfg=cfg.sigma,
        training_cfg=cfg.training,
        sampler_cfg=cfg.sampler,
        loss_cfg=cfg.get("loss", None),
    )
    _make_model_backward_compatible()
    # Training checkpoints contain OmegaConf metadata in addition to tensor
    # weights.  PyTorch >=2.6 defaults to weights_only=True, which rejects
    # that trusted local metadata during standalone inference.
    loaded_weights = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # Compact ``model_weights.pt`` files intentionally contain a single EMA
    # state_dict; retain compatibility with full Lightning checkpoints.
    state_dict_key = (
        "ema_weights"
        if use_ema_weights and "ema_weights" in loaded_weights
        else "state_dict"
    )
    state_dict = loaded_weights[state_dict_key]

    # Add a model. prefix to the keys
    keys = list(state_dict.keys())
    first_key = keys[0]
    if not first_key.startswith("model."):
        for key in keys:
            state_dict[f"model.{key}"] = state_dict.pop(key)

    # Replace batch_norm keys with norm
    keys = list(state_dict.keys())
    for key in keys:
        if "batch_norm" in key:
            new_key = key.replace("batch_norm", "norm")
            state_dict[new_key] = state_dict.pop(key)

    for key, value in state_dict.items():
        if value.isnan().any():
            raise ValueError("Values cannot be nan")

    # The compact checkpoint still stores the Lightning module state dict.
    # Load it exactly like a full training checkpoint and return the module;
    # this return block was accidentally lost while extracting IFMDock.
    with contextlib.nullcontext():
        model.load_state_dict(state_dict, strict=True)
        if freeze:
            model.freeze()

    return model
