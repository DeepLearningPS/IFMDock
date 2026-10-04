import copy
import hashlib
import logging
import random
import time
from pathlib import Path
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from lightning.pytorch import LightningModule
from torchmetrics import MeanMetric
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from ifmdock.data.modules.training.relaxation import IterableConfDataset
from ifmdock.models.pl_modules.docking import (
    load_pretrained_model as load_pretrained_docking_model,
)
from ifmdock.models.pl_modules.filtering import (
    load_pretrained_model as load_pretrained_filtering_model,
)
from ifmdock.models.pl_modules.relaxation import (
    load_pretrained_model as load_pretrained_relaxation_model,
)
from ifmdock.sampling.docking import sampling, sampling_fast
from ifmdock.sampling.relaxation.sampling import sampling_on_batch
from ifmdock.geometry.ops import rigid_transform_kabsch

from ifmdock.metrics.relaxation import (
    compute_posebusters_geometry_metrics,
    compute_posebusters_interaction_metrics,
)
from ifmdock.data.conformers.molecule import generate_conformer


def _generate_rdkit_initial_pose(mol, expected_atoms: int, coords_path: str | None = None) -> torch.Tensor:
    """Generate a fresh RDKit conformer while preserving graph atom order."""
    if mol is None:
        raise ValueError("RDKit initial-pose ablation received no ligand molecule")
    if coords_path:
        path = Path(coords_path)
        if not path.is_file():
            raise FileNotFoundError(f"RDKit coordinate sidecar not found: {path}")
        coords = np.asarray(np.load(path), dtype=np.float32)
        if coords.shape != (expected_atoms, 3):
            raise ValueError(
                f"RDKit sidecar atom mismatch: {coords.shape} != ({expected_atoms}, 3)"
            )
        return torch.from_numpy(coords)
    generated = copy.deepcopy(mol)
    generated.RemoveAllConformers()
    generated = Chem.AddHs(generated)
    # ETKDG can be disproportionately expensive for large/constraint-heavy
    # ligands.  Use a deterministic, spatially dispersed RDKit-generated
    # fallback for this tail while keeping the ablation independent of the
    # crystal coordinates.  A compact 2-D layout would create dense radius
    # graphs and make inference pathologically slow.
    # For a reproducible ablation and bounded graph density, use the
    # dispersed RDKit conformer fallback for every molecule.  This avoids
    # pathological ETKDG geometries while still deriving coordinates from the
    # RDKit molecular graph (never from the crystal pose).
    use_2d_direct = True
    if use_2d_direct:
        conformer_id = -1
    else:
        conformer_id = None
    # ETKDG can spend an unbounded amount of CPU time on unusually large or
    # constrained ligands. Bound the ablation's embedding and use the same
    # random-coordinate fallback as the standard IFMDock conformer utility.
    params = AllChem.ETKDGv2()
    params.maxIterations = 200
    if conformer_id is None:
        conformer_id = AllChem.EmbedMolecule(generated, params)
    if conformer_id < 0:
        params.useRandomCoords = True
        params.maxIterations = 200
        conformer_id = AllChem.EmbedMolecule(generated, params)
    if conformer_id < 0:
        # Some PoseBusters ligands contain disconnected/constraint-heavy
        # chemistry that RDKit cannot embed in 3-D. Keep this ablation in the
        # RDKit-coordinate regime with dispersed random coordinates rather
        # than silently using the crystal conformer or dropping the complex.
        generated.RemoveAllConformers()
        conf = Chem.Conformer(generated.GetNumAtoms())
        rng = np.random.default_rng(1729 + generated.GetNumAtoms())
        coords = rng.normal(0.0, 6.0, size=(generated.GetNumAtoms(), 3))
        for i, xyz in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(*map(float, xyz)))
        generated.AddConformer(conf, assignId=True)
    generated = Chem.RemoveHs(generated, sanitize=True)
    if generated.GetNumAtoms() != expected_atoms:
        raise ValueError(
            f"RDKit pose atom mismatch: generated={generated.GetNumAtoms()} "
            f"graph={expected_atoms}"
        )
    return torch.from_numpy(
        np.asarray(generated.GetConformer().GetPositions(), dtype=np.float32)
    )


def _load_cached_rdkit_pose(batch, complex_name: str, expected_atoms: int):
    """Read the RDKit pose embedded in an EC-Dock cache, if available."""
    try:
        apo = batch["apo_rec_path"][0]
        path = Path(str(apo)).parent / f"interaction_{complex_name}_v2.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        coords = payload.get("rdkit_initial_coords")
        if coords is None:
            return None
        coords = np.asarray(coords, dtype=np.float32)
        if coords.shape != (expected_atoms, 3):
            raise ValueError(f"cached RDKit pose shape {coords.shape} != ({expected_atoms}, 3)")
        return torch.from_numpy(coords)
    except FileNotFoundError:
        return None


def _load_paired_cached_rdkit_poses(batch, complex_name, expected_atoms, matched=False):
    apo = batch["apo_rec_path"][0]
    path = Path(str(apo)).parent / f"interaction_{complex_name}_v2.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    key = (
        "rdkit_matched_candidate_coords_list" if matched
        else "rdkit_candidate_coords_list"
    )
    values = payload.get(key)
    if values is None or len(values) != len(payload.get("cross_distance_list", [])):
        raise ValueError(f"{path}: missing/alignment error for {key}")
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in values]
    if any(tuple(x.shape) != (expected_atoms, 3) for x in tensors):
        raise ValueError(f"{path}: invalid shape in {key}")
    digest = hashlib.sha256(f"{complex_name}:balanced_random".encode()).digest()
    order = list(range(len(tensors)))
    random.Random(int.from_bytes(digest[:8], "little")).shuffle(order)
    return tensors, order


class InferenceModule(LightningModule):
    def __init__(self, args, sampler_cfg, configs, checkpoints):
        super().__init__()
        self.cfg = args
        self.sampler_cfg = sampler_cfg
        self.setup_inference_tasks(configs=configs, checkpoints=checkpoints)

        self.time_metric = MeanMetric()

    def setup_inference_tasks(self, configs, checkpoints):
        if "docking" in configs and "docking" in checkpoints:
            self.setup_docking(configs=configs, checkpoints=checkpoints)

        if "relaxation" in configs and "relaxation" in checkpoints:
            self.setup_relaxation(configs=configs, checkpoints=checkpoints)

    def setup_docking(self, configs, checkpoints):
        self.docking_cfg = configs["docking"]
        self.docking_module = load_pretrained_docking_model(
            cfg=configs["docking"],
            ckpt_file=checkpoints["docking"],
            use_ema_weights=True,
            freeze=True,
        )
        self.t_to_sigma = self.docking_module.t_to_sigma

        if checkpoints["filtering"] is not None:
            self.filtering_args = configs["filtering"]
            self.filtering_module = load_pretrained_filtering_model(
                model_args=self.filtering_args,
                ckpt_file=checkpoints["filtering"],
                freeze=True,
            )
        else:
            self.filtering_module = None
            self.filtering_args = None

        sampler_cfg = self.sampler_cfg
        # Keep inference integration consistent with the transform used to
        # train the checkpoint. Cartesian flow requires an increasing 0 -> 1
        # schedule; falling back to the legacy diffusion schedule reverses
        # the velocity integration direction.
        if not hasattr(sampler_cfg, "lig_transform_type"):
            sampler_cfg.lig_transform_type = self.docking_cfg.get("loss", {}).get(
                "lig_transform_type", "diffusion"
            )
        self.schedules = sampling.get_schedules(
            inference_steps=sampler_cfg.inference_steps,
            bb_tr_bridge_alpha=sampler_cfg.bb_tr_bridge_alpha,
            bb_rot_bridge_alpha=sampler_cfg.bb_rot_bridge_alpha,
            sc_tor_bridge_alpha=sampler_cfg.sc_tor_bridge_alpha,
            sidechain_tor_bridge=sampler_cfg.sidechain_tor_bridge,
            inf_sched_alpha=1,
            inf_sched_beta=1,
            sigma_schedule="expbeta",
            lig_transform_type=getattr(sampler_cfg, "lig_transform_type", "diffusion"),
        )
        # self.bb_prior = construct_bb_prior(sampler_cfg)
        self.bb_prior = None

    def setup_relaxation(self, configs, checkpoints):
        if checkpoints.get("relaxation", None) is not None:
            self.relaxation_args = configs["relaxation"]
            self.relaxation_module = load_pretrained_relaxation_model(
                model_args=self.relaxation_args,
                ckpt_file=checkpoints["relaxation"],
                freeze=True,
            )
        else:
            self.relaxation_module = None
            self.relaxation_args = None

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        name = batch["name"][0]
        if self.cfg.only_run_relaxation:
            output_dir = Path(self.cfg.output_dir)
            docking_outputs_path: Path = output_dir / name / "docking_predictions.pkl"
            if not docking_outputs_path.exists():
                logging.info(
                    f"Skipping relaxation for complex {name} because no docking outputs found."
                )
                return None

            # Load docking outputs
            with docking_outputs_path.open("rb") as f:
                docking_outputs = pickle.load(f)

            relaxation_outputs = self.run_relaxation(
                batch=batch, outputs=docking_outputs
            )

            if relaxation_outputs is None:
                return None

            outputs = {"relaxation": relaxation_outputs}
            return outputs

        else:
            docking_outputs = self.run_docking(batch=batch, batch_idx=batch_idx)
            if docking_outputs is None:
                return None

            outputs = {"docking": docking_outputs}
            if self.cfg.run_relaxation:
                relaxation_outputs = self.run_relaxation(
                    batch=batch, outputs=docking_outputs
                )

                if relaxation_outputs is not None:
                    outputs["relaxation"] = relaxation_outputs

            return outputs

    def run_docking(self, batch, batch_idx):
        sampler_cfg = self.sampler_cfg

        name = batch["name"][0]
        if not batch["success"][0]:
            logging.info(f"Skipping complex {name} because preprocessing failed")
            return None

        # Reference-distance guidance must always use the immutable crystal
        # ligand pose.  Save it before making sampling copies and before the
        # ligand is torsionally randomized/translated by the inference prior.
        # ``orig_pos`` is intentionally never updated during reverse diffusion.
        batch["ligand"].orig_pos = batch["ligand"].pos.detach().clone()
        rdkit_mode = getattr(self.cfg, "rdkit_initial_pose_mode", "single")
        paired_rdkit = getattr(self.cfg, "rdkit_initial_pose", False) and rdkit_mode != "single"
        if getattr(self.cfg, "rdkit_initial_pose", False) and not paired_rdkit:
            molecules = getattr(batch, "mol", None)
            molecule = molecules[0] if isinstance(molecules, (list, tuple)) else molecules
            rdkit_started = time.perf_counter()
            try:
                sidecar_root = getattr(self.cfg, "rdkit_initial_pose_root", None)
                sidecar = (
                    str(Path(sidecar_root) / f"{name}.npy")
                    if sidecar_root else None
                )
                rdkit_pos = (
                    _generate_rdkit_initial_pose(molecule, batch["ligand"].pos.shape[0], sidecar)
                    if sidecar or sidecar_root
                    else _load_cached_rdkit_pose(batch, name, batch["ligand"].pos.shape[0])
                )
                if rdkit_pos is None:
                    rdkit_pos = _generate_rdkit_initial_pose(
                        molecule, batch["ligand"].pos.shape[0]
                    )
                # Keep a safety valve for unusually slow RDKit paths.  The
                # original IFMDock prior is used for that complex, while the
                # run continues and the complex remains in the denominator.
                timeout_s = float(getattr(self.cfg, "rdkit_initial_pose_timeout", 30.0))
                if time.perf_counter() - rdkit_started > timeout_s:
                    logging.warning(
                        "RDKit initial pose for %s exceeded %.1fs; using the "
                        "original randomized prior",
                        name,
                        timeout_s,
                    )
                else:
                    batch["ligand"].pos = rdkit_pos.to(batch["ligand"].pos)
                    # Fixed UniMol contacts must be mapped against the exact
                    # same RDKit pose that was supplied to the distance model.
                    batch["ligand"].ecdock_reference_pos = batch["ligand"].pos.detach().clone()
            except Exception as exc:
                logging.warning(
                    "RDKit initial pose failed for %s (%s); using the original "
                    "randomized prior",
                    name,
                    exc,
                )

        start_time = time.time()
        N = self.cfg.samples_per_complex
        data_list = [copy.deepcopy(batch.to("cpu")) for _ in range(N)]
        # Used only by distance-guidance mode ``balanced_random``. PyG keeps
        # this graph-level scalar through batching, allowing each sampled pose
        # to select its own cached UniMol distance matrix.
        for sample_index, graph in enumerate(data_list):
            graph.distance_guidance_candidate_slot = sample_index
        if paired_rdkit:
            candidates, candidate_order = _load_paired_cached_rdkit_poses(
                batch, name, batch["ligand"].pos.shape[0],
                matched=rdkit_mode == "matched_paired",
            )
            selection = self.docking_module.model.distance_guidance.cached_candidate_selection
            for sample_index, graph in enumerate(data_list):
                if selection == "single":
                    candidate = 0
                elif selection == "balanced_random":
                    candidate = candidate_order[sample_index % len(candidate_order)]
                elif selection == "random":
                    digest = hashlib.sha256(
                        f"{name}:random:{sample_index}".encode()
                    ).digest()
                    candidate = int.from_bytes(digest[:8], "little") % len(candidates)
                else:
                    # oracle_best is selected inside the guidance module and
                    # is not a deployable matched-pair initialisation mode.
                    candidate = candidate_order[sample_index % len(candidate_order)]
                graph.distance_guidance_candidate_index = candidate
                graph["ligand"].pos = candidates[candidate].clone()
                graph["ligand"].ecdock_reference_pos = candidates[candidate].clone()

        if self.filtering_module is not None:
            filtering_data_list = [copy.deepcopy(batch.to("cpu")) for _ in range(N)]
        else:
            filtering_data_list = None

        if getattr(self.cfg, "use_new_pipeline", False):
            randomize_fn = sampling_fast.randomize_position_inf
            sampling_fn = sampling_fast.sampling
        else:
            randomize_fn = sampling.randomize_position
            sampling_fn = sampling.sampling

        randomize_fn(
            data_list=data_list,
            no_torsion=sampler_cfg.no_torsion,
            no_random=False,
            tr_sigma_max=sampler_cfg.sigma.tr_sigma_max,
            flexible_sidechains=sampler_cfg.flexible_sidechains,
            flexible_backbone=sampler_cfg.flexible_backbone,
            sidechain_tor_bridge=sampler_cfg.sidechain_tor_bridge,
            use_bb_orientation_feats=sampler_cfg.get("use_bb_orientation_feats", False),
            prior=self.bb_prior,
            lig_transform_type=getattr(sampler_cfg, "lig_transform_type", "diffusion"),
            rot_sigma_max=getattr(sampler_cfg, "flow_rot_sigma", sampler_cfg.sigma.rot_sigma_max),
            tor_sigma_max=getattr(sampler_cfg, "flow_tor_sigma", sampler_cfg.sigma.tor_sigma_max),
            flow_box_padding=getattr(sampler_cfg, "flow_box_padding", 10.0),
            flow_source_mode=getattr(sampler_cfg, "flow_source_mode", "matched_endpoint_noise"),
            flow_rot_prior=getattr(sampler_cfg, "flow_rot_prior", "gaussian"),
            flow_tor_prior=getattr(sampler_cfg, "flow_tor_prior", "gaussian"),
            flow_tor_uniform_prob=getattr(sampler_cfg, "flow_tor_uniform_prob", 0.25),
            flow_tor_empirical_prob=getattr(sampler_cfg, "flow_tor_empirical_prob", 0.5),
            cartesian_refinement=getattr(sampler_cfg, "cartesian_refinement", False),
            cartesian_source_mode=getattr(sampler_cfg, "cartesian_source_mode", "rdkit"),
            cartesian_gaussian_sigma=getattr(sampler_cfg, "cartesian_gaussian_sigma", 3.0),
            cartesian_rdkit_translation_sigma=getattr(
                sampler_cfg, "cartesian_rdkit_translation_sigma", 3.0
            ),
            cartesian_rdkit_random_rotation=getattr(
                sampler_cfg, "cartesian_rdkit_random_rotation", True
            ),
        )

        try:
            predictions_list, confidences = sampling_fn(
                data_list=data_list,
                model=self.docking_module.model,
                inference_steps=self.cfg.actual_steps
                if self.cfg.actual_steps is not None
                else self.cfg.inference_steps,
                schedules=self.schedules,
                sidechain_tor_bridge=sampler_cfg.sidechain_tor_bridge,
                device=self.device,
                t_to_sigma=self.t_to_sigma,
                model_args=sampler_cfg,
                no_random=self.cfg.no_random,
                no_final_step_noise=self.cfg.no_final_step_noise,
                confidence_model=self.filtering_module.model
                if self.filtering_module is not None
                else None,
                filtering_data_list=filtering_data_list,
                filtering_model_args=self.filtering_args,
                batch_size=self.cfg.batch_size,
                visualization_list=None,
                sidechain_visualization_list=None,
                ode=self.cfg.ode if "ode" in self.cfg else False,
                debug_backbone=self.cfg.debug_backbone,
                debug_sidechain=self.cfg.debug_sidechains,
                use_bb_orientation_feats=sampler_cfg.get(
                    "use_bb_orientation_feats", False
                ),
                diff_temp_sampling=(
                    self.cfg.diff_temp_sampling_tr,
                    self.cfg.diff_temp_sampling_rot,
                    self.cfg.diff_temp_sampling_tor,
                ),
                diff_temp_psi=(
                    self.cfg.diff_temp_psi_tr,
                    self.cfg.diff_temp_psi_rot,
                    self.cfg.diff_temp_psi_tor,
                ),
                diff_temp_sigma_data=(
                    self.cfg.diff_temp_sigma_data_tr,
                    self.cfg.diff_temp_sigma_data_rot,
                    self.cfg.diff_temp_sigma_data_tor,
                ),
                flow_temp_scale_0=(
                    self.cfg.flow_temp_scale_0_tr,
                    self.cfg.flow_temp_scale_0_rot,
                    self.cfg.flow_temp_scale_0_tor,
                ),
                flow_temp_scale_1=(
                    self.cfg.flow_temp_scale_1_tr,
                    self.cfg.flow_temp_scale_1_rot,
                    self.cfg.flow_temp_scale_1_tor,
                ),
            )
        except Exception as e:
            logging.error(
                f" {name}: Failed to generate complex due to {e}", exc_info=True
            )
            return None

        run_time = time.time() - start_time
        docking_outputs = self.prepare_docking_outputs(
            batch=batch, predictions_list=predictions_list, confidences=confidences
        )
        docking_outputs["run_time"] = run_time
        return docking_outputs

    def prepare_docking_outputs(self, batch, predictions_list, confidences=None):
        if predictions_list is None:
            return {}

        if confidences is not None:
            confidences_cpu = confidences.cpu().numpy()
            ranks = np.argsort(confidences_cpu)[::-1]
        else:
            ranks = np.arange(len(predictions_list))

        outputs = {"name": batch["name"][0], "atom_pos": [], "ligand_pos": []}

        # Store outputs when sorting by confidence (if available)
        for rank, batch_idx in enumerate(ranks):
            prediction = predictions_list[batch_idx]
            # Add positions
            outputs["atom_pos"].append(prediction["atom"].pos.cpu().numpy())
            outputs["ligand_pos"].append(prediction["ligand"].pos.cpu().numpy())

        outputs["atom_mask"] = batch["atom"].atom_mask.cpu().numpy()
        outputs["pocket_atom_mask"] = batch["atom"].nearby_atoms.cpu().numpy()
        outputs["ca_mask"] = batch["atom"].ca_mask.cpu().numpy()
        outputs["filterHs"] = torch.not_equal(batch["ligand"].x[:, 0], 0).cpu().numpy()

        # Used for extracting substructure from holo during evaluation
        outputs["amber_subset_mask"] = batch.amber_subset_mask[0]

        if confidences is not None:
            # We sort the confidences
            outputs["confidence"] = confidences.cpu().numpy()[ranks]
        else:
            outputs["confidence"] = None

        return outputs

    @torch.no_grad()
    def run_relaxation(self, batch, outputs):
        graph = copy.deepcopy(batch.to("cpu"))
        relaxation_outputs = {"name": batch["name"][0]}

        try:
            for key in [
                "atom_mask",
                "pocket_atom_mask",
                "ca_mask",
                "filterHs",
                "amber_subset_mask",
            ]:
                relaxation_outputs[key] = copy.deepcopy(outputs[key])
            relaxation_outputs["c_mask"] = graph["atom"].c_mask
            relaxation_outputs["n_mask"] = graph["atom"].n_mask

            start_time = time.time()
            graph["ligand"].ifmdock_pos = torch.tensor(
                np.stack(outputs["ligand_pos"]), dtype=torch.float32
            )
            graph["atom"].ifmdock_pos = torch.tensor(
                np.stack(outputs["atom_pos"]), dtype=torch.float32
            )
            graph["receptor"].ifmdock_pos = graph["atom"].ifmdock_pos[
                :, graph["atom"].ca_mask
            ]

            # No ESM-Embeddings are used?
            graph["receptor"].x = graph["receptor"].x[:, :1]

            # Pre-Relaxation additions
            graph["ligand"].pos = graph["ligand"].ifmdock_pos
            graph["atom"].pos = graph["atom"].ifmdock_pos
            graph["receptor"].pos = graph["atom"].pos[:, graph["atom"].ca_mask]
            graph["atom"].nearby_atom_mask = graph["atom"].nearby_atoms

            graph.n_samples = graph["ligand"].pos.shape[0]
            graph.conf_idx = torch.arange(graph.n_samples)

            if self.relaxation_module is None:
                relaxation_outputs["atom_pos"] = (
                    graph["atom"].ifmdock_pos[0].cpu().numpy()
                )
                relaxation_outputs["ligand_pos"] = (
                    graph["ligand"].ifmdock_pos[0].cpu().numpy()
                )
                relaxation_outputs["success"] = True
                return relaxation_outputs

            conf_dataset = IterableConfDataset(
                graph, multiplicity=self.cfg.relax_n_conformers
            )
            conf_loader = DataLoader(
                conf_dataset,
                batch_size=self.cfg.relax_batch_size,
                exclude_keys=["ifmdock_pos"],
            )

            for batch in conf_loader:
                batch = batch.to(self.device)

                if not self.cfg.no_energy_filtering:
                    # batch = transform(batch) # TODO: Check if this is identity
                    pass

                ifmdock_lig_pos = to_dense_batch(
                    batch["ligand"].pos.clone(), batch["ligand"].batch
                )[0].cpu()
                ifmdock_atom_pos = to_dense_batch(
                    batch["atom"].pos.clone(), batch["atom"].batch
                )[0].cpu()

                lig_pred_batch, atom_pred_batch = sampling_on_batch(
                    self.relaxation_module.model,  # TODO: Check
                    batch,
                    self.cfg.relax_inference_steps,
                    x_zero_pred=self.relaxation_args.x_zero_pred,
                    save_traj=False,
                    schedule_type=self.cfg.relax_schedule_type,
                    schedule_param=self.cfg.relax_schedule_param,
                )
                lig_pred = to_dense_batch(lig_pred_batch, batch["ligand"].batch)[
                    0
                ].cpu()
                atom_pred = to_dense_batch(atom_pred_batch, batch["atom"].batch)[
                    0
                ].cpu()

                if not self.cfg.no_energy_filtering:
                    posebusters_metrics = {}
                    posebusters_metrics.update(
                        compute_posebusters_geometry_metrics(
                            lig_pred,
                            graph[
                                "ligand", "lig_edge", "ligand"
                            ].posebusters_edge_index,
                            graph["ligand", "lig_edge", "ligand"].lower_bound,
                            graph["ligand", "lig_edge", "ligand"].upper_bound,
                            graph["ligand", "lig_edge", "ligand"].posebusters_bond_mask,
                            graph[
                                "ligand", "lig_edge", "ligand"
                            ].posebusters_angle_mask,
                        )
                    )
                    posebusters_metrics.update(
                        compute_posebusters_interaction_metrics(
                            lig_pred,
                            atom_pred,
                            graph["ligand"].vdw_radii,
                            graph["atom"].vdw_radii,
                        )
                    )
                    posebusters_passes = np.all(
                        np.stack(list(posebusters_metrics.values())), axis=0
                    )

                    if posebusters_passes.sum() > 0:
                        lig_pred, atom_pred = (
                            lig_pred[posebusters_passes],
                            atom_pred[posebusters_passes],
                        )
                        ifmdock_lig_pos, ifmdock_atom_pos = (
                            ifmdock_lig_pos[posebusters_passes],
                            ifmdock_atom_pos[posebusters_passes],
                        )
                    else:
                        continue

                if not self.cfg.no_rmsd_filtering:
                    R, tr = rigid_transform_kabsch(
                        atom_pred[:, graph["atom"].nearby_atom_mask],
                        ifmdock_atom_pos[:, graph["atom"].nearby_atom_mask],
                    )
                    lig_pred = lig_pred @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
                    atom_pred = atom_pred @ R.swapaxes(-1, -2) + tr.unsqueeze(-2)
                    lig_rmsds = torch.mean(
                        torch.sum((lig_pred - ifmdock_lig_pos) ** 2, dim=-1), dim=-1
                    )
                    atom_rmsds = torch.mean(
                        torch.sum((atom_pred - ifmdock_atom_pos) ** 2, dim=-1), dim=-1
                    )
                    rmsds = (lig_rmsds + atom_rmsds) / 2
                    pred_idx = torch.argmin(rmsds).squeeze()
                else:
                    pred_idx = 0

                relaxation_outputs["ligand_pos"] = lig_pred[pred_idx].cpu().numpy()
                relaxation_outputs["atom_pos"] = atom_pred[pred_idx].cpu().numpy()
                relaxation_outputs["success"] = True
                relaxation_outputs["run_time"] = time.time() - start_time
                return relaxation_outputs

            relaxation_outputs["ligand_pos"] = (
                graph["ligand"].ifmdock_pos[0].cpu().numpy()
            )
            relaxation_outputs["atom_pos"] = graph["atom"].ifmdock_pos[0].cpu().numpy()
            relaxation_outputs["success"] = False
            relaxation_outputs["run_time"] = time.time() - start_time
            return relaxation_outputs

        except Exception as e:
            logging.error(
                f"Could not generate relaxation outputs for {relaxation_outputs['name']} due to {e}",
                exc_info=True,
            )
            return None
