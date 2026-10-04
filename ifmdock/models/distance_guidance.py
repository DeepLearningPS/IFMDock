"""EC-Dock/UniMol distance guidance for rigid-pocket IFMDock.

The adapter deliberately keeps the two graph index spaces separate until the
last operation.  UniMol receives padded *per-complex* ligand/pocket tensors;
the resulting local contact indices are then mapped back through the PyG batch
node indices.  Consequently no edge can join atoms from different complexes.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
import pickle
import random
import sys
from argparse import Namespace
from pathlib import Path

import torch

from ifmdock import get_atom_order_metadata, get_index_mapping_metadata
from torch import nn


EC_DISTANCE_ROOT = Path(os.environ.get(
    "IFMDOCK_DISTANCE_MODEL_ROOT",
    Path(__file__).resolve().parents[1] / "utils" / "third_party" / "distance_model",
))
EC_MAIN_RUNNING_ROOT = EC_DISTANCE_ROOT.parent
EC_CHECKPOINT = EC_DISTANCE_ROOT / "premodel" / "best.pt"


def _ec_namespace(data_dir: Path, recycling: int) -> Namespace:
    return Namespace(
        data=str(data_dir), task="docking", loss="docking", arch="docking",
        seed=42, conf_size=1, dist_threshold=8.0, max_pocket_atoms=400,
        recycling=recycling, max_seq_len=512, mol_pooler_dropout=0.2,
        pocket_pooler_dropout=0.2, mol_dropout=0.2, pocket_dropout=0.2,
        finetune_mol_model=None, finetune_pocket_model=None, mode="infer",
        debug_shapes=False, cross_distance_loss_weight=1.0,
        holo_distance_loss_weight=1.0, coord_loss_weight=1.0,
        prmsd_loss_weight=0.1,
    )


class ECDockDistanceGuidance(nn.Module):
    """Pretrained EC-Dock distance head plus safe batched contact rebuilding."""

    def __init__(
        self,
        cutoff: float = 4.5,
        loss_weight: float = 1.0,
        cross_loss_weight: float = 0.1,
        holo_loss_weight: float = 0.05,
        target_max_distance: float = 8.0,
        checkpoint: str | None = None,
        data_dir: str | None = None,
        recycling: int = 1,
        freeze_encoder: bool = False,
        update_parameters: bool = True,
        use_atom_edges: bool = True,
        use_residue_edges: bool = False,
        include_ifmdock_atom_edges: bool = True,
        include_ifmdock_residue_edges: bool = True,
        source: str = "predicted",
        split_distance: float = 3.5,
        extension_distance: float = 2.0,
        type_residue_edges: bool = False,
        cached_candidate_selection: str = "single",
        cached_mapping_tolerance: float = 0.05,
        require_explicit_index_mapping: bool = True,
        cached_top_n_per_ligand: int | None = None,
        cached_shell_top_n_per_ligand: int | None = None,
        cached_outer_cutoff: float | None = None,
        cached_outer_top_n_per_ligand: int | None = None,
        use_extension_edges: bool = True,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.loss_weight = float(loss_weight)
        self.cross_loss_weight = float(cross_loss_weight)
        self.holo_loss_weight = float(holo_loss_weight)
        self.target_max_distance = float(target_max_distance)
        self.update_parameters = bool(update_parameters)
        self.use_atom_edges = bool(use_atom_edges)
        self.use_residue_edges = bool(use_residue_edges)
        self.include_ifmdock_atom_edges = bool(include_ifmdock_atom_edges)
        self.include_ifmdock_residue_edges = bool(include_ifmdock_residue_edges)
        self.source = str(source)
        self.split_distance = float(split_distance)
        self.extension_distance = float(extension_distance)
        self.type_residue_edges = bool(type_residue_edges)
        self.cached_candidate_selection = str(cached_candidate_selection)
        self.cached_mapping_tolerance = float(cached_mapping_tolerance)
        self.require_explicit_index_mapping = bool(require_explicit_index_mapping)
        self.cached_top_n_per_ligand = (
            None if cached_top_n_per_ligand is None
            else int(cached_top_n_per_ligand)
        )
        self.cached_shell_top_n_per_ligand = (
            None if cached_shell_top_n_per_ligand is None
            else int(cached_shell_top_n_per_ligand)
        )
        self.cached_outer_cutoff = (
            None if cached_outer_cutoff is None else float(cached_outer_cutoff)
        )
        self.cached_outer_top_n_per_ligand = (
            None if cached_outer_top_n_per_ligand is None
            else int(cached_outer_top_n_per_ligand)
        )
        self.use_extension_edges = bool(use_extension_edges)
        if (
            self.cached_top_n_per_ligand is not None
            and self.cached_shell_top_n_per_ligand is not None
        ):
            raise ValueError("global Top-N and shell Top-N are mutually exclusive")
        for label, value in (
            ("cached_top_n_per_ligand", self.cached_top_n_per_ligand),
            ("cached_shell_top_n_per_ligand", self.cached_shell_top_n_per_ligand),
            ("cached_outer_top_n_per_ligand", self.cached_outer_top_n_per_ligand),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive or null")
        if (self.cached_outer_cutoff is None) != (
            self.cached_outer_top_n_per_ligand is None
        ):
            raise ValueError(
                "cached_outer_cutoff and cached_outer_top_n_per_ligand must be set together"
            )
        if self.cached_outer_cutoff is not None and self.cached_outer_cutoff <= self.cutoff:
            raise ValueError("cached_outer_cutoff must be greater than cutoff")
        self._fixed_contact_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.source not in {"predicted", "reference", "ecdock_cache"}:
            raise ValueError(f"Unknown distance-guidance source: {self.source}")
        if self.cached_candidate_selection not in {
            "single", "random", "balanced_random", "oracle_best"
        }:
            raise ValueError(
                "cached_candidate_selection must be single, random, "
                "balanced_random, or oracle_best"
            )

        # The reference-distance ablation is an upper-bound experiment.  It
        # constructs EC-Dock contacts directly from the crystal pose and must
        # not instantiate the ~2 GB UniMol distance model at all.
        if self.source in {"reference", "ecdock_cache"}:
            self.ec_model = None
            self.mol_dictionary = None
            self.pocket_dictionary = None
            self.update_parameters = False
            return

        for root in (str(EC_MAIN_RUNNING_ROOT), str(EC_DISTANCE_ROOT)):
            if root not in sys.path:
                sys.path.insert(0, root)
        # Registration is performed by this import before setup_task().
        import generalmodels  # noqa: F401
        from Distance_model.basemodels import tasks

        dictionary_dir = Path(data_dir) if data_dir else EC_DISTANCE_ROOT / "example_data"
        args = _ec_namespace(dictionary_dir, recycling)
        task = tasks.setup_task(args)
        self.ec_model = task.build_model(args)
        state_path = Path(checkpoint) if checkpoint else EC_CHECKPOINT
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        incompatible = self.ec_model.load_state_dict(state["model"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "EC-Dock distance checkpoint mismatch: "
                f"missing={incompatible.missing_keys[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        self.mol_dictionary = task.dictionary
        self.pocket_dictionary = task.pocket_dictionary
        # ``distance_only=True`` deliberately bypasses EC-Dock's legacy
        # coordinate-refinement and pRMSD heads.  They would otherwise remain
        # unused on every rank and make DDP's reducer track a large, pointless
        # conditional parameter set.  The two encoders, concat decoder and
        # cross-distance head stay trainable and are the complete distance
        # prediction path used by stage 2.
        for module_name in (
            "coord_decoder",
            "coord_delta_project",
            "prmsd_project",
            "concat_gbf",
            "concat_gbf_proj",
        ):
            for parameter in getattr(self.ec_model, module_name).parameters():
                parameter.requires_grad_(False)
        if freeze_encoder:
            for parameter in self.ec_model.parameters():
                parameter.requires_grad_(False)
            for parameter in self.ec_model.cross_distance_project.parameters():
                parameter.requires_grad_(True)
        if not self.update_parameters and self.ec_model is not None:
            # Fixed-guidance experiment: preserve the original pretrained
            # EC-Dock predictor exactly and keep it out of Adam/EMA/DDP.
            self.ec_model.requires_grad_(False)
            self.ec_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.update_parameters and self.ec_model is not None:
            # Calling model.train() recursively must not re-enable dropout in
            # the frozen distance predictor.
            self.ec_model.eval()
        return self

    @staticmethod
    def _element_symbols(encoded_atomic_numbers: torch.Tensor) -> list[str]:
        # IFMDock categorical atomic-number features use atomic_number - 1.
        symbols = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I",
        ]
        return [symbols[int(value)] if 0 <= int(value) < len(symbols) else "[UNK]" for value in encoded_atomic_numbers]

    @staticmethod
    def _with_special_tokens(tokens, coords, dictionary):
        token_ids = torch.tensor(
            [dictionary.bos()] + [dictionary.index(token) for token in tokens] + [dictionary.eos()],
            device=coords.device,
            dtype=torch.long,
        )
        padded_coords = torch.cat(
            [coords.new_zeros((1, 3)), coords, coords.new_zeros((1, 3))], dim=0
        )
        return token_ids, padded_coords

    @staticmethod
    def _pad(items, pad_value=0):
        max_len = max(item.shape[0] for item in items)
        output = items[0].new_full((len(items), max_len), pad_value)
        for i, item in enumerate(items):
            output[i, : item.shape[0]] = item
        return output

    @staticmethod
    def _pad_coords(items):
        max_len = max(item.shape[0] for item in items)
        output = items[0].new_zeros((len(items), max_len, 3))
        for i, item in enumerate(items):
            output[i, : item.shape[0]] = item
        return output

    def forward(self, data):
        lig_batch, atom_batch = data["ligand"].batch, data["atom"].batch
        graph_count = int(data.num_graphs)
        if self.source == "reference":
            return self._forward_reference(data, lig_batch, atom_batch, graph_count)
        if self.source == "ecdock_cache":
            return self._forward_ecdock_cache(data, lig_batch, atom_batch, graph_count)
        lig_tokens, atom_tokens, lig_coords, atom_coords = [], [], [], []
        local_node_indices = []
        cross_targets, holo_targets = [], []

        reference_ligand = getattr(data["ligand"], "orig_pos", data["ligand"].pos)
        reference_atoms = getattr(data["atom"], "orig_aligned_apo_pos", data["atom"].pos)
        for graph_id in range(graph_count):
            lig_idx = torch.where(lig_batch == graph_id)[0]
            atom_idx = torch.where(atom_batch == graph_id)[0]
            if len(lig_idx) == 0 or len(atom_idx) == 0:
                raise ValueError(f"empty ligand or pocket in graph {graph_id}")
            lig_token, lig_coord = self._with_special_tokens(
                self._element_symbols(data["ligand"].x[lig_idx, 0]),
                data["ligand"].pos[lig_idx], self.mol_dictionary,
            )
            atom_token, atom_coord = self._with_special_tokens(
                self._element_symbols(data["atom"].x[atom_idx, 1]),
                data["atom"].pos[atom_idx], self.pocket_dictionary,
            )
            lig_tokens.append(lig_token)
            atom_tokens.append(atom_token)
            lig_coords.append(lig_coord)
            atom_coords.append(atom_coord)
            local_node_indices.append((lig_idx, atom_idx))
            reference_lig = reference_ligand[lig_idx]
            cross_targets.append(torch.cdist(reference_lig, reference_atoms[atom_idx]))
            holo_targets.append(torch.cdist(reference_lig, reference_lig))

        mol_tokens, pocket_tokens = self._pad(lig_tokens), self._pad(atom_tokens)
        mol_coords, pocket_coords = self._pad_coords(lig_coords), self._pad_coords(atom_coords)
        mol_distance = torch.cdist(mol_coords, mol_coords)
        pocket_distance = torch.cdist(pocket_coords, pocket_coords)
        mol_edge_type = mol_tokens.unsqueeze(-1) * len(self.mol_dictionary) + mol_tokens.unsqueeze(-2)
        pocket_edge_type = pocket_tokens.unsqueeze(-1) * len(self.pocket_dictionary) + pocket_tokens.unsqueeze(-2)
        context = nullcontext() if self.update_parameters else torch.no_grad()
        with context:
            cross_prediction, holo_prediction = self.ec_model(
                mol_tokens, mol_distance, mol_coords, mol_edge_type,
                pocket_tokens, pocket_distance, pocket_coords, pocket_edge_type,
                distance_only=True,
            )

        cross_losses, holo_losses, edge_parts, type_parts = [], [], [], []
        for graph_id, (lig_idx, atom_idx) in enumerate(local_node_indices):
            predicted_cross = cross_prediction[
                graph_id, 1 : len(lig_idx) + 1, 1 : len(atom_idx) + 1
            ]
            cross_target = cross_targets[graph_id]
            # EC-Dock / UniMol Docking V2 supervises only positive cross
            # distances below ``dist_threshold`` (8 Å by default), with MSE.
            cross_valid = (cross_target > 0) & (cross_target < self.target_max_distance)
            if cross_valid.any():
                cross_losses.append(
                    torch.nn.functional.mse_loss(
                        predicted_cross[cross_valid], cross_target[cross_valid]
                    )
                )

            predicted_holo = holo_prediction[
                graph_id, 1 : len(lig_idx) + 1, 1 : len(lig_idx) + 1
            ]
            holo_target = holo_targets[graph_id]
            # EC-Dock has no distance cutoff on its holo-ligand head.  The
            # diagonal is zero and is excluded by the original >0 mask.
            holo_valid = holo_target > 0
            if holo_valid.any():
                holo_losses.append(
                    torch.nn.functional.smooth_l1_loss(
                        predicted_holo[holo_valid], holo_target[holo_valid]
                    )
                )

            # Rebuild exactly the same EC-Dock contact relations as reference
            # mode, changing only the source of the ligand--protein distance
            # matrix.  Direct contacts are 12/13 and protein-neighbour
            # extensions are 16; reverse 14/15/17 are assigned by the score
            # network when it flips the directed edge.
            direct_mask = (predicted_cross > 0) & (predicted_cross <= self.cutoff)
            local_lig, local_atom = torch.where(direct_mask)
            if local_lig.numel() > 0:
                edge_parts.append(
                    torch.stack([lig_idx[local_lig], atom_idx[local_atom]])
                )
                type_parts.append(torch.where(
                    predicted_cross[local_lig, local_atom] < self.split_distance,
                    torch.full_like(local_lig, 12),
                    torch.full_like(local_lig, 13),
                ))

                protein_distance = torch.cdist(
                    data["atom"].pos[atom_idx], data["atom"].pos[atom_idx]
                )
                protein_near = protein_distance <= self.extension_distance
                extension_mask = (
                    direct_mask.to(torch.float32)
                    @ protein_near.to(torch.float32)
                    > 0
                ) & ~direct_mask
                extension_lig, extension_atom = torch.where(extension_mask)
                if extension_lig.numel() > 0:
                    edge_parts.append(torch.stack([
                        lig_idx[extension_lig], atom_idx[extension_atom]
                    ]))
                    type_parts.append(torch.full(
                        (extension_lig.numel(),),
                        16,
                        dtype=torch.long,
                        device=predicted_cross.device,
                    ))

        data.distance_guidance_la_edge_index = (
            torch.cat(edge_parts, dim=1)
            if edge_parts
            else torch.empty((2, 0), dtype=torch.long, device=data["ligand"].pos.device)
        )
        data.distance_guidance_la_edge_type = (
            torch.cat(type_parts)
            if type_parts
            else torch.empty(
                (0,), dtype=torch.long, device=data["ligand"].pos.device
            )
        )
        # Lift predicted ligand--atom contacts to ligand--residue contacts via
        # the explicit PyG atom->receptor relation.  Build a lookup by source
        # index instead of assuming that relation columns happen to be sorted.
        guided_la = data.distance_guidance_la_edge_index
        atom_receptor = data["atom", "receptor"].edge_index
        atom_to_receptor = torch.full(
            (data["atom"].num_nodes,), -1, dtype=torch.long, device=guided_la.device
        )
        atom_to_receptor[atom_receptor[0].long()] = atom_receptor[1].long()
        if guided_la.numel() > 0:
            guided_receptor = atom_to_receptor[guided_la[1]]
            if (guided_receptor < 0).any():
                raise ValueError("A distance-guided protein atom has no receptor mapping")
            guided_lr = torch.unique(
                torch.stack([guided_la[0], guided_receptor]), dim=1
            )
            if not torch.equal(
                data["ligand"].batch[guided_lr[0]],
                data["receptor"].batch[guided_lr[1]],
            ):
                raise ValueError("Distance-guided ligand--receptor edge crosses graphs")
        else:
            guided_lr = torch.empty(
                (2, 0), dtype=torch.long, device=guided_la.device
            )
        data.distance_guidance_lr_edge_index = guided_lr
        zero = cross_prediction.sum() * 0.0
        data.distance_guidance_cross_loss = (
            torch.stack(cross_losses).mean() if cross_losses else zero
        )
        data.distance_guidance_holo_loss = (
            torch.stack(holo_losses).mean() if holo_losses else zero
        )
        data.distance_guidance_loss = (
            self.cross_loss_weight * data.distance_guidance_cross_loss
            + self.holo_loss_weight * data.distance_guidance_holo_loss
        )
        return data.distance_guidance_loss

    @staticmethod
    def _batch_metadata(value, graph_id):
        if isinstance(value, (list, tuple)):
            value = value[graph_id]
            while isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            return value
        if torch.is_tensor(value) and value.ndim > 0:
            return value.reshape(-1)[graph_id]
        return value

    def _select_cached_candidate(
        self, payload, complex_name: str, candidate_slot: int | None = None,
        candidate_index: int | None = None,
    ) -> int:
        count = len(payload["cross_distance_list"])
        if count == 0:
            raise ValueError(f"{complex_name}: empty cross_distance_list")
        if candidate_index is not None:
            candidate_index = int(candidate_index)
            if candidate_index < 0 or candidate_index >= count:
                raise ValueError(f"{complex_name}: invalid candidate index {candidate_index}/{count}")
            return candidate_index
        if self.cached_candidate_selection == "single":
            return 0
        if self.cached_candidate_selection == "random":
            if candidate_slot is None:
                # Training graphs have no pose slot. Worker RNGs are seeded by
                # PyTorch, so this draws a new cached UniMol candidate on every
                # dataset visit (normally once per epoch) while remaining
                # reproducible for a fixed run seed.
                return int(torch.randint(count, (1,)).item())
            # Inference graphs carry a pose slot. Select independently per
            # pose, but deterministically so resumed/sharded evaluation gives
            # exactly the same result.
            digest = hashlib.sha256(
                f"{complex_name}:random:{int(candidate_slot)}".encode()
            ).digest()
            return int.from_bytes(digest[:8], "little") % count
        if self.cached_candidate_selection == "balanced_random":
            if candidate_slot is None:
                raise ValueError(
                    f"{complex_name}: balanced_random requires a per-pose candidate slot"
                )
            # Deterministically shuffle all available UniMol candidates, then
            # cycle through that permutation. Thus 40 poses and 10 candidates
            # use every matrix exactly four times while remaining reproducible.
            digest = hashlib.sha256(
                f"{complex_name}:balanced_random".encode()
            ).digest()
            order = list(range(count))
            random.Random(int.from_bytes(digest[:8], "little")).shuffle(order)
            return order[int(candidate_slot) % count]
        predicted = payload.get("coords_predict_list")
        holo = payload.get("holo_coords_list")
        if predicted is None or holo is None or len(predicted) != count:
            raise ValueError(f"{complex_name}: oracle_best needs predicted/holo coordinates")
        rmsds = [
            float(torch.as_tensor((pred - ref) ** 2).sum(-1).mean().sqrt())
            for pred, ref in zip(predicted, holo)
        ]
        return int(torch.tensor(rmsds).argmin())

    def _load_fixed_contacts(
        self,
        complex_name: str,
        apo_path: str,
        ligand_reference: torch.Tensor,
        atom_reference: torch.Tensor,
        candidate_slot: int | None = None,
        candidate_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interaction_path = Path(apo_path).parent / f"interaction_{complex_name}_v2.pkl"
        if not interaction_path.is_file():
            raise FileNotFoundError(
                f"EC-Dock fixed-distance cache not found: {interaction_path}. "
                "Generate it with the EC-Dock/UniMol interface before training."
            )
        with interaction_path.open("rb") as handle:
            payload = pickle.load(handle)
        candidate = self._select_cached_candidate(
            payload, complex_name, candidate_slot=candidate_slot,
            candidate_index=candidate_index,
        )
        cache_key = f"{complex_name}:{self.cached_candidate_selection}:{candidate}"
        if cache_key in self._fixed_contact_cache:
            return self._fixed_contact_cache[cache_key]
        cross_distance = torch.as_tensor(
            payload["cross_distance_list"][candidate], dtype=torch.float32
        )
        cached_ligand = torch.as_tensor(
            payload["holo_coords_list"][candidate], dtype=torch.float32
        )
        cached_pocket = torch.as_tensor(
            payload["pocket_coords_list"][candidate], dtype=torch.float32
        )
        if cross_distance.shape != (len(cached_ligand), len(cached_pocket)):
            raise ValueError(
                f"{interaction_path}: distance shape {tuple(cross_distance.shape)} "
                f"does not match ligand/pocket coordinates"
            )

        def unique_nearest(source, target, label):
            # Absolute PDB coordinates can be O(100 A); float32 cdist's
            # quadratic expansion suffers cancellation and can report ~0.06 A
            # for identical coordinates.  Map in float64 before thresholding.
            distances = torch.cdist(source.double(), target.double())
            nearest_distance, nearest = distances.min(dim=1)
            valid = nearest_distance <= self.cached_mapping_tolerance
            # Reproduce EC-Dock's coordinate-to-PDB mapping: unmatched cached
            # pocket columns are discarded.  Duplicate mappings are unsafe.
            mapped = nearest[valid]
            if mapped.unique().numel() != mapped.numel():
                raise ValueError(f"{complex_name}: non-unique {label} coordinate mapping")
            return valid, nearest

        # When the distance model was run from an RDKit conformer, its ligand
        # coordinates intentionally differ from the crystal/randomized pose
        # used by IFMDock inference.  The SDF atom order is preserved, so map
        # ligand rows by their stable atom order rather than by incompatible
        # 3-D coordinates.  Protein rows still use the coordinate bijection.
        index_mapping = get_index_mapping_metadata(payload)
        if self.require_explicit_index_mapping and not (
            isinstance(index_mapping, dict)
            and index_mapping.get("version") == 1
            and index_mapping.get("method") == "chemical_identity_v1"
            and index_mapping.get("runtime_coordinate_mapping") is False
        ):
            raise ValueError(
                f"{complex_name}: a validated explicit atom-index permutation "
                "is required; coordinate-based runtime mapping is disabled"
            )
        explicit_ligand = (
            index_mapping.get("ligand_to_pyg")
            if isinstance(index_mapping, dict) else None
        )
        if explicit_ligand is not None:
            ligand_map = torch.as_tensor(explicit_ligand, dtype=torch.long)
            if len(ligand_map) != len(cached_ligand) or (
                ligand_map.numel() and (
                    int(ligand_map.min()) < 0 or int(ligand_map.max()) >= len(ligand_reference)
                )
            ):
                raise ValueError(f"{complex_name}: invalid explicit ligand-to-PyG mapping")
            ligand_valid = torch.ones(len(cached_ligand), dtype=torch.bool)
        elif payload.get("rdkit_initial_coords") is not None:
            if len(cached_ligand) != len(ligand_reference):
                raise ValueError(
                    f"{complex_name}: RDKit cache ligand atoms {len(cached_ligand)} "
                    f"!= graph atoms {len(ligand_reference)}"
                )
            ligand_valid = torch.ones(len(cached_ligand), dtype=torch.bool)
            ligand_map = torch.arange(len(cached_ligand), dtype=torch.long)
        else:
            ligand_valid, ligand_map = unique_nearest(
                cached_ligand, ligand_reference.cpu(), "ligand"
            )
        if not ligand_valid.all():
            raise ValueError(
                f"{complex_name}: only {int(ligand_valid.sum())}/{len(ligand_valid)} "
                "cached ligand atoms map to the IFMDock graph"
            )
        explicit_protein = (
            index_mapping.get("protein_to_pyg")
            if isinstance(index_mapping, dict) else None
        )
        order_info = get_atom_order_metadata(payload)
        direct_order = (
            isinstance(order_info, dict)
            and order_info.get("version") == 1
            and len(cached_pocket) == len(atom_reference)
            and torch.allclose(
                cached_pocket.double(), atom_reference.cpu().double(),
                atol=self.cached_mapping_tolerance, rtol=0,
            )
        )
        if explicit_protein is not None:
            pocket_map = torch.as_tensor(explicit_protein, dtype=torch.long)
            if len(pocket_map) != len(cached_pocket) or (
                pocket_map.numel() and (
                    int(pocket_map.min()) < 0 or int(pocket_map.max()) >= len(atom_reference)
                )
            ):
                raise ValueError(f"{complex_name}: invalid explicit protein-to-PyG mapping")
            pocket_valid = torch.ones(len(cached_pocket), dtype=torch.bool)
        elif direct_order:
            pocket_valid = torch.ones(len(cached_pocket), dtype=torch.bool)
            pocket_map = torch.arange(len(cached_pocket), dtype=torch.long)
        else:
            # Backward compatibility for legacy EC-Dock caches.  Newly
            # finalized caches use the direct, canonical IFMDock order above.
            pocket_valid, pocket_map = unique_nearest(
                cached_pocket, atom_reference.cpu(), "protein"
            )
        direct_mask = (
            (cross_distance > 0)
            & (cross_distance <= self.cutoff)
            & pocket_valid.unsqueeze(0)
        )
        if self.cached_shell_top_n_per_ligand is not None:
            core_mask = direct_mask & (cross_distance < self.split_distance)
            shell_mask = direct_mask & ~core_mask
            selected_shell = torch.zeros_like(shell_mask)
            for ligand_index in range(len(shell_mask)):
                candidates = torch.where(shell_mask[ligand_index])[0]
                if candidates.numel():
                    count = min(self.cached_shell_top_n_per_ligand, candidates.numel())
                    nearest = torch.topk(
                        cross_distance[ligand_index, candidates], count,
                        largest=False,
                    ).indices
                    selected_shell[ligand_index, candidates[nearest]] = True
            direct_mask = core_mask | selected_shell
        elif self.cached_top_n_per_ligand is not None:
            selected = torch.zeros_like(direct_mask)
            for ligand_index in range(len(direct_mask)):
                candidates = torch.where(direct_mask[ligand_index])[0]
                if candidates.numel():
                    count = min(self.cached_top_n_per_ligand, candidates.numel())
                    nearest = torch.topk(
                        cross_distance[ligand_index, candidates], count,
                        largest=False,
                    ).indices
                    selected[ligand_index, candidates[nearest]] = True
            direct_mask = selected
        cached_lig, cached_atom = torch.where(direct_mask)
        local_lig = ligand_map[cached_lig]
        local_atom = pocket_map[cached_atom]
        edge_parts = [torch.stack([local_lig, local_atom])]
        type_parts = [torch.where(
            cross_distance[cached_lig, cached_atom] < self.split_distance,
            torch.full_like(local_lig, 12), torch.full_like(local_lig, 13),
        )]

        # Optional third distance band.  Keep all direct <=cutoff contacts,
        # then add only the nearest K predicted contacts per ligand atom from
        # (cutoff, outer_cutoff].  Type 16/17 is reserved for this weaker,
        # directed context relation when geometric extension edges are off.
        if self.cached_outer_cutoff is not None:
            outer_mask = (
                (cross_distance > self.cutoff)
                & (cross_distance <= self.cached_outer_cutoff)
                & pocket_valid.unsqueeze(0)
            )
            selected_outer = torch.zeros_like(outer_mask)
            for ligand_index in range(len(outer_mask)):
                candidates = torch.where(outer_mask[ligand_index])[0]
                if candidates.numel():
                    count = min(
                        self.cached_outer_top_n_per_ligand, candidates.numel()
                    )
                    nearest = torch.topk(
                        cross_distance[ligand_index, candidates], count,
                        largest=False,
                    ).indices
                    selected_outer[ligand_index, candidates[nearest]] = True
            outer_ligand, outer_atom = torch.where(selected_outer)
            if outer_ligand.numel():
                edge_parts.append(torch.stack([
                    ligand_map[outer_ligand], pocket_map[outer_atom]
                ]))
                type_parts.append(torch.full_like(outer_ligand, 16))

        # EC-Dock type 16: for each ligand atom, extend direct contacts to
        # other mapped protein atoms within extension_distance.
        direct_local = torch.zeros(
            (len(ligand_reference), len(atom_reference)), dtype=torch.bool
        )
        direct_local[local_lig, local_atom] = True
        if self.use_extension_edges:
            protein_near = torch.cdist(atom_reference.cpu(), atom_reference.cpu()) <= self.extension_distance
            extension = (
                direct_local.float() @ protein_near.float() > 0
            ) & ~direct_local
            extension_lig, extension_atom = torch.where(extension)
            if extension_lig.numel():
                edge_parts.append(torch.stack([extension_lig, extension_atom]))
                type_parts.append(torch.full_like(extension_lig, 16))
        result = (torch.cat(edge_parts, dim=1), torch.cat(type_parts))
        self._fixed_contact_cache[cache_key] = result
        return result

    def _forward_ecdock_cache(self, data, lig_batch, atom_batch, graph_count):
        """Attach EC-Dock contacts once per complex and reuse them for every x_t."""
        device = data["ligand"].pos.device
        names = getattr(data, "name")
        apo_paths = getattr(data, "apo_rec_path")
        centers = getattr(data, "original_center", None)
        if centers is not None:
            centers = centers.reshape(graph_count, 3)
        reference_ligand = getattr(data["ligand"], "orig_pos", None)
        cached_ligand_reference = getattr(
            data["ligand"], "ecdock_reference_pos", None
        )
        cached_atom_reference = getattr(data["atom"], "ecdock_reference_pos", None)
        candidate_slots = getattr(data, "distance_guidance_candidate_slot", None)
        candidate_indices = getattr(data, "distance_guidance_candidate_index", None)
        if reference_ligand is None and cached_ligand_reference is None:
            raise ValueError("Fixed EC-Dock guidance requires ligand.orig_pos")
        edge_parts, type_parts = [], []
        for graph_id in range(graph_count):
            lig_idx = torch.where(lig_batch == graph_id)[0]
            atom_idx = torch.where(atom_batch == graph_id)[0]
            center = centers[graph_id].detach().cpu() if centers is not None else torch.zeros(3)
            ligand_abs = (
                cached_ligand_reference[lig_idx].detach().cpu()
                if cached_ligand_reference is not None
                else reference_ligand[lig_idx].detach().cpu() + center
            )
            atom_abs = (
                cached_atom_reference[atom_idx].detach().cpu()
                if cached_atom_reference is not None
                else data["atom"].pos[atom_idx].detach().cpu() + center
            )
            local_edge, local_type = self._load_fixed_contacts(
                str(self._batch_metadata(names, graph_id)),
                str(self._batch_metadata(apo_paths, graph_id)),
                ligand_abs,
                atom_abs,
                candidate_slot=(
                    int(self._batch_metadata(candidate_slots, graph_id))
                    if candidate_slots is not None else None
                ),
                candidate_index=(
                    int(self._batch_metadata(candidate_indices, graph_id))
                    if candidate_indices is not None else None
                ),
            )
            edge_parts.append(torch.stack([
                lig_idx[local_edge[0]].to(device), atom_idx[local_edge[1]].to(device)
            ]))
            type_parts.append(local_type.to(device))
        data.distance_guidance_la_edge_index = torch.cat(edge_parts, dim=1)
        data.distance_guidance_la_edge_type = torch.cat(type_parts)

        # Residue lifting remains available for a later ablation, but experiment
        # E keeps use_residue_edges=false so atom and residue guidance are not
        # changed simultaneously.
        atom_receptor = data["atom", "receptor"].edge_index
        atom_to_receptor = torch.full(
            (data["atom"].num_nodes,), -1, dtype=torch.long, device=device
        )
        atom_to_receptor[atom_receptor[0].long()] = atom_receptor[1].long()
        guided_receptor = atom_to_receptor[data.distance_guidance_la_edge_index[1]]
        if (guided_receptor < 0).any():
            raise ValueError("A fixed EC-Dock protein atom has no receptor mapping")
        data.distance_guidance_lr_edge_index = torch.unique(
            torch.stack([data.distance_guidance_la_edge_index[0], guided_receptor]),
            dim=1,
        )
        zero = data["ligand"].pos.sum() * 0.0
        data.distance_guidance_cross_loss = zero
        data.distance_guidance_holo_loss = zero
        data.distance_guidance_loss = zero
        return zero

    def _forward_reference(self, data, lig_batch, atom_batch, graph_count):
        """Build EC-Dock 12--17 contacts from the crystallographic pose.

        Edge indices are global PyG node indices.  Direct contacts are split
        at 3.5/4.5 A (types 12/13); type 16 extends each ligand atom's direct
        protein contacts to previously uncontacted protein atoms within 2 A.
        Reverse types 14/15/17 are encoded later when IFMDock flips the edge.
        """
        device = data["ligand"].pos.device
        ref_ligand = getattr(data["ligand"], "orig_pos", None)
        # In rigid-pocket training atom.pos is never noised and is already in
        # the same centered frame as ligand.orig_pos.  The cached orig_* atom
        # tensors are intentionally not re-centered by PocketTransform when
        # the protein is rigid, so using them here would create an invalid
        # cross-frame distance matrix.
        ref_atoms = data["atom"].pos
        if ref_ligand is None or ref_atoms is None:
            raise ValueError("Reference-distance guidance requires crystal ligand/protein coordinates")

        edge_parts, type_parts = [], []
        direct_counts, extension_counts = [], []
        for graph_id in range(graph_count):
            lig_idx = torch.where(lig_batch == graph_id)[0]
            atom_idx = torch.where(atom_batch == graph_id)[0]
            if lig_idx.numel() == 0 or atom_idx.numel() == 0:
                raise ValueError(f"empty ligand or pocket in graph {graph_id}")
            cross_distance = torch.cdist(ref_ligand[lig_idx], ref_atoms[atom_idx])
            local_lig, local_atom = torch.where(
                (cross_distance > 0) & (cross_distance <= self.cutoff)
            )
            if local_lig.numel() == 0:
                raise ValueError(
                    f"reference contact graph {graph_id} has no edge within {self.cutoff} A"
                )
            direct_edges = torch.stack([lig_idx[local_lig], atom_idx[local_atom]])
            direct_types = torch.where(
                cross_distance[local_lig, local_atom] < self.split_distance,
                torch.full_like(local_lig, 12),
                torch.full_like(local_lig, 13),
            )
            edge_parts.append(direct_edges)
            type_parts.append(direct_types)
            direct_counts.append(local_lig.numel())

            # Match EC-Dock's split3_5_extend, but vectorize all ligand atoms.
            # The literal EC-Dock Python loop performs many GPU-synchronizing
            # scalar tests and can leave the other DDP ranks waiting for a
            # graph with a larger ligand.  Boolean matrix multiplication is
            # exactly equivalent: direct[L,P] @ near[P,P] identifies protein
            # neighbours of every ligand atom's direct-contact set.
            protein_distance = torch.cdist(ref_atoms[atom_idx], ref_atoms[atom_idx])
            direct_mask = (cross_distance > 0) & (cross_distance <= self.cutoff)
            protein_near = protein_distance <= self.extension_distance
            extension_mask = (
                direct_mask.to(torch.float32) @ protein_near.to(torch.float32) > 0
            ) & ~direct_mask
            extension_lig, extension_atom = torch.where(extension_mask)
            if extension_lig.numel() > 0:
                edge_parts.append(torch.stack([
                    lig_idx[extension_lig], atom_idx[extension_atom]
                ]))
                type_parts.append(torch.full(
                    (extension_lig.numel(),), 16, dtype=torch.long, device=device
                ))
            extension_counts.append(extension_lig.numel())

        edge_index = torch.cat(edge_parts, dim=1)
        edge_type = torch.cat(type_parts)
        if not torch.equal(lig_batch[edge_index[0]], atom_batch[edge_index[1]]):
            raise ValueError("Reference-distance edge crosses PyG graphs")
        data.distance_guidance_la_edge_index = edge_index
        data.distance_guidance_la_edge_type = edge_type

        # Lift every reference ligand--protein-atom contact to the residue
        # containing that protein atom.  PyG batches concatenate each node
        # type independently, so the atom and receptor indices must be mapped
        # through the explicit atom->receptor relation; they must never be
        # inferred from ordering or residue lengths.  Multiple atom contacts
        # can map to the same ligand--residue pair and are deliberately
        # deduplicated here so residue message strength does not scale with
        # the number of atoms in a residue.
        atom_receptor = data["atom", "receptor"].edge_index
        atom_to_receptor = torch.full(
            (data["atom"].num_nodes,), -1, dtype=torch.long, device=device
        )
        atom_to_receptor[atom_receptor[0].long()] = atom_receptor[1].long()
        guided_receptor = atom_to_receptor[edge_index[1]]
        if (guided_receptor < 0).any():
            raise ValueError("A reference-guided protein atom has no receptor mapping")
        residue_pairs = torch.stack([edge_index[0], guided_receptor])
        residue_edge_index, residue_inverse = torch.unique(
            residue_pairs, dim=1, return_inverse=True
        )
        if self.type_residue_edges:
            # A ligand atom can contact several atoms in the same residue.
            # Keep one ligand--residue edge and give direct contacts priority
            # over shell contacts, which in turn have priority over extension
            # contacts.  Atom relation IDs 12/13/16 are deliberately ordered
            # by that priority.  Residue relations use an independent ID
            # range: 18/19/22 forward and 20/21/23 reverse.
            residue_atom_type = torch.full(
                (residue_edge_index.shape[1],),
                16,
                dtype=torch.long,
                device=device,
            )
            residue_atom_type.scatter_reduce_(
                0, residue_inverse, edge_type, reduce="amin", include_self=True
            )
            residue_edge_type = torch.where(
                residue_atom_type == 12,
                torch.full_like(residue_atom_type, 18),
                torch.where(
                    residue_atom_type == 13,
                    torch.full_like(residue_atom_type, 19),
                    torch.full_like(residue_atom_type, 22),
                ),
            )
        else:
            residue_edge_type = torch.zeros(
                residue_edge_index.shape[1], dtype=torch.long, device=device
            )
        if not torch.equal(
            lig_batch[residue_edge_index[0]],
            data["receptor"].batch[residue_edge_index[1]],
        ):
            raise ValueError("Reference-distance ligand--residue edge crosses PyG graphs")
        data.distance_guidance_lr_edge_index = residue_edge_index
        data.distance_guidance_lr_edge_type = residue_edge_type
        data.distance_guidance_direct_counts = torch.tensor(
            direct_counts, dtype=torch.long, device=device
        )
        data.distance_guidance_extension_counts = torch.tensor(
            extension_counts, dtype=torch.long, device=device
        )
        zero = data["ligand"].pos.sum() * 0.0
        data.distance_guidance_cross_loss = zero
        data.distance_guidance_holo_loss = zero
        data.distance_guidance_loss = zero
        return zero
