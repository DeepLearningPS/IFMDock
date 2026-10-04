import os
import random
import csv
import json
import copy

from typing import Optional, Any
from dataclasses import dataclass, fields

import torch
import pickle
import numpy as np
from torch_geometric.data import Dataset
from torch.utils.data import Sampler

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from torch_geometric.loader import DataLoader

from ifmdock.data.constants import AVAILABLE_DATASETS
from ifmdock.data.parse.base import read_strings_from_txt
from ifmdock.data.transforms.docking import construct_transform
from ifmdock.data.feature.canonical_torsion import attach_canonical_torsions


def _chemical_torsion_periods(complex_graph):
    """Infer symmetry periods from atom identity, never from coordinates.

    ``breakTies=False`` assigns the same canonical rank to graph-equivalent
    substituents.  A two- or three-fold set around the rotating endpoint gives
    pi or 2pi/3 periodicity; all other bonds conservatively retain 2pi.
    """
    n_torsions = int(complex_graph["ligand"].edge_mask.sum())
    periods = torch.full((n_torsions,), 2 * np.pi, dtype=torch.float32)
    mol = getattr(complex_graph, "mol", None)
    if mol is None or mol.GetNumAtoms() != int(complex_graph["ligand"].num_nodes):
        return periods
    try:
        from rdkit import Chem
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        edge_index = complex_graph["ligand", "lig_bond", "ligand"].edge_index
        selected = edge_index[:, complex_graph["ligand"].edge_mask].T.tolist()
        for i, (fixed, rotating) in enumerate(selected):
            neighbor_ranks = [
                ranks[nbr.GetIdx()]
                for nbr in mol.GetAtomWithIdx(int(rotating)).GetNeighbors()
                if nbr.GetIdx() != int(fixed)
            ]
            multiplicity = max(
                (neighbor_ranks.count(rank) for rank in set(neighbor_ranks)),
                default=1,
            )
            if multiplicity in (2, 3):
                periods[i] = 2 * np.pi / multiplicity
    except Exception:
        # A conservative 2pi period is always chemically valid.
        pass
    return periods


def _attach_canonical_torsions(complex_graph):
    """Attach one deterministic i-j-k-l definition per directed rotatable bond."""
    mol = getattr(complex_graph, "mol", None)
    n_atoms = int(complex_graph["ligand"].num_nodes)
    edge_index = complex_graph["ligand", "lig_bond", "ligand"].edge_index
    selected = edge_index[:, complex_graph["ligand"].edge_mask].T.tolist()
    if mol is None or mol.GetNumAtoms() != n_atoms:
        raise ValueError("canonical torsions require an atom-aligned RDKit molecule")
    from rdkit import Chem
    ranks = list(Chem.CanonicalRankAtoms(mol, includeChirality=True, breakTies=True))
    tuples = []
    for j, k in selected:
        left = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        right = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
        if not left or not right:
            raise ValueError(f"rotatable bond {j}-{k} lacks canonical reference atoms")
        i = min(left, key=lambda atom: (ranks[atom], atom))
        l = min(right, key=lambda atom: (ranks[atom], atom))
        tuples.append((i, j, k, l))
    complex_graph["ligand"].canonical_torsion_index = torch.tensor(
        tuples, dtype=torch.long
    ).T.contiguous() if tuples else torch.empty((4, 0), dtype=torch.long)
    complex_graph["ligand"].canonical_tor_period = _chemical_torsion_periods(complex_graph)


def _attach_torsion_reliability(complex_graph):
    """Chemistry-only confidence for each directed rotatable-bond target."""
    n_torsions = int(complex_graph["ligand"].edge_mask.sum())
    weights = torch.ones(n_torsions, dtype=torch.float32)
    mol = getattr(complex_graph, "mol", None)
    if mol is None or mol.GetNumAtoms() != int(complex_graph["ligand"].num_nodes):
        complex_graph["ligand"].tor_reliability = weights
        return
    selected = complex_graph["ligand", "lig_bond", "ligand"].edge_index[
        :, complex_graph["ligand"].edge_mask
    ].T.tolist()
    periods = _chemical_torsion_periods(complex_graph)
    for idx, (_, rotating) in enumerate(selected):
        atom = mol.GetAtomWithIdx(int(rotating))
        heavy_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1)
        # Terminal/methyl-like rotors and symmetry-ambiguous representatives
        # carry valid but weak orientation supervision.
        if heavy_neighbors <= 1:
            weights[idx] *= 0.25
        if float(periods[idx]) < (2 * np.pi - 1e-4):
            weights[idx] *= 0.25
    complex_graph["ligand"].tor_reliability = weights


class ListDataset(Dataset):
    def __init__(self, list, transform=None):
        super().__init__(root=None, transform=transform)
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int):
        return self.data_list[idx]


class DistributedWeightedSampler(Sampler):
    """Deterministic replacement sampling shared and sharded across DDP ranks."""

    def __init__(self, weights, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_samples = (len(self.weights) + self.world_size - 1) // self.world_size
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        )
        return iter(indices[self.rank : self.total_size : self.world_size].tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _restore_aligned_rdkit_mol(graph, dataset_root, complex_name):
    """Restore the chemistry graph omitted from compact PyG caches."""
    cached_mol = getattr(graph, "mol", None)
    if (cached_mol is not None
            and cached_mol.GetNumAtoms() == int(graph["ligand"].num_nodes)):
        return cached_mol
    if dataset_root is None:
        raise ValueError(f"{complex_name}: dataset root is required for canonical torsions")
    from rdkit import Chem
    sdf = os.path.join(dataset_root, complex_name, f"{complex_name}_ligand.sdf")
    supplier = Chem.SDMolSupplier(sdf, sanitize=True, removeHs=True)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"{complex_name}: RDKit failed to restore {sdf}")
    if mol.GetNumAtoms() != int(graph["ligand"].num_nodes):
        raise ValueError(
            f"{complex_name}: SDF/PyG atom count mismatch "
            f"({mol.GetNumAtoms()} != {graph['ligand'].num_nodes})"
        )
    # The cached ligand graph and source SDF must have exactly the same bonds
    # in exactly the same atom identity order; coordinates are never used.
    graph_edges = graph["ligand", "lig_bond", "ligand"].edge_index.T.tolist()
    if any(mol.GetBondBetweenAtoms(int(a), int(b)) is None for a, b in graph_edges):
        raise ValueError(f"{complex_name}: SDF/PyG chemical bond identity mismatch")
    graph.mol = mol
    return mol


@dataclass
class DockingDataConfig:
    dataset: str

    # all paths
    cache_path: str
    split_train: str
    split_val: str
    cluster_file: Optional[str] = None

    require_ligand: bool = True
    exclude_complexes: tuple[str, ...] = ()
    interaction_cache_root: Optional[str] = None
    # Atomically select one paired RDKit/matched/UniMol candidate before the
    # stochastic flow transform.  The selected index is forwarded to distance
    # guidance so conformer and cross-distance matrix can never diverge.
    paired_distance_conformer: bool = False
    paired_candidate_matching_rmsd_max: Optional[float] = None
    prepare_paired_torsion_metadata: bool = True
    # Flow-L poses used as x_0 by Cartesian refinement.  The directory must
    # contain train/predictions/<complex>/docking_predictions.pkl and the
    # analogous val tree.
    stage2_initial_pose_root: Optional[str] = None
    # Optional flattened pool of qualified Cartesian-relaxation starts.  A
    # directory is resolved as <directory>/<split>_manifest.json.
    stage2_pose_manifest: Optional[str] = None
    hard_overlap_cache: Optional[str] = None
    hard_overlap_sampling: bool = False
    hard_overlap_weights: tuple[float, float, float, float] = (1.0, 1.5, 2.0, 3.0)

    # some cache cfg
    limit_complexes: int = 0
    complexes_per_cluster: int = 10
    multiplicity: int = 1

    # validation inference
    run_val_inference: bool = False
    num_inference_complexes: int = 500
    # Train-only fine-tuning must not eagerly build and preload a duplicate
    # validation Dataset on every DDP rank.
    disable_validation: bool = False

    # data loading
    batch_size: int = 4
    num_workers: int = 1
    num_dataloader_workers: int = 26
    pin_memory: bool = False
    dataloader_drop_last: bool = False
    # Keep immutable graph/cache templates in host RAM. This avoids rank-local
    # torch.load stalls that otherwise surface later as NCCL timeouts.
    preload_to_memory: bool = False

    @classmethod
    def from_dict(cls, data: dict):
        # Create an instance of the data class from a dictionary
        return cls(
            **{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        )


class DockingDataset(Dataset):
    def __init__(
        self,
        transform: callable,
        split_path: str,
        dataset: str = "pdbbind",
        cache_path: str = "data/cache",  # Full cache path
        cluster_file: Optional[str] = None,
        complexes_per_cluster: int = 1,
        limit_complexes: int = 0,
        multiplicity: int = 1,
        num_workers: int = 1,
        require_ligand: bool = False,
        exclude_complexes=(),
        interaction_cache_root: Optional[str] = None,
        paired_distance_conformer: bool = False,
        paired_candidate_matching_rmsd_max: Optional[float] = None,
        prepare_paired_torsion_metadata: bool = True,
        stage2_initial_pose_root: Optional[str] = None,
        stage2_pose_manifest: Optional[str] = None,
        stage2_split: str = "train",
        hard_overlap_cache: Optional[str] = None,
        preload_to_memory: bool = False,
    ):
        super().__init__(transform=transform)
        self.dataset_name = dataset
        self.split_path = split_path
        self.cache_path = cache_path
        self.limit_complexes = limit_complexes
        self.multiplicity = multiplicity
        self.num_workers = num_workers
        self.cluster_file = cluster_file
        self.complexes_per_cluster = complexes_per_cluster
        self.require_ligand = require_ligand
        self.exclude_complexes = set(exclude_complexes or ())
        self.interaction_cache_root = interaction_cache_root
        self.paired_distance_conformer = paired_distance_conformer
        self.paired_candidate_matching_rmsd_max = paired_candidate_matching_rmsd_max
        self.prepare_paired_torsion_metadata = bool(prepare_paired_torsion_metadata)
        self.stage2_initial_pose_root = stage2_initial_pose_root
        self.stage2_pose_manifest = stage2_pose_manifest
        self.stage2_split = stage2_split
        self.hard_overlap_cache = hard_overlap_cache
        self.preload_to_memory = bool(preload_to_memory)

        if not self.check_processed_inputs():
            raise ValueError("Inputs must be processed before running training.")

        self.gather_processed_inputs()
        self.prepare_clustering_dicts()
        self.subsample_clusters()
        self.sampling_weights = self._load_sampling_weights()
        self._graph_memory_cache = {}
        self._interaction_memory_cache = {}
        if self.preload_to_memory:
            self._preload_inputs()

    def _preload_inputs(self):
        """Load immutable templates before DDP starts; clone graphs per sample."""
        for name in self.complex_files:
            graph_path = os.path.join(self.cache_path, f"{name}.pt")
            self._graph_memory_cache[name] = torch.load(
                graph_path, weights_only=False, map_location="cpu"
            )
            if self.paired_distance_conformer:
                complex_name = name.removeprefix("heterograph-").rsplit("-", 1)[0]
                interaction_path = os.path.join(
                    self.interaction_cache_root, complex_name,
                    f"interaction_{complex_name}_v2.pkl",
                )
                with open(interaction_path, "rb") as handle:
                    self._interaction_memory_cache[complex_name] = pickle.load(handle)
        print(
            f"Preloaded {len(self._graph_memory_cache)} graphs and "
            f"{len(self._interaction_memory_cache)} interaction caches into host RAM",
            flush=True,
        )

    def _manifest_path(self):
        if self.stage2_pose_manifest is None:
            return None
        path = self.stage2_pose_manifest.format(split=self.stage2_split)
        if os.path.isdir(path):
            path = os.path.join(path, f"{self.stage2_split}_manifest.json")
        return path

    def _load_sampling_weights(self):
        if self.hard_overlap_cache is None:
            return None
        if hasattr(self, "stage2_samples"):
            raise ValueError(
                "hard-overlap replacement sampling is intentionally disabled for "
                "a qualified-pose manifest; it would undo per-complex balancing"
            )
        with open(self.hard_overlap_cache) as handle:
            payload = json.load(handle)
        overlap_by_complex = payload["complexes"]
        thresholds = payload["thresholds"]
        level_weights = payload.get("level_weights", [1.0, 1.5, 2.0, 3.0])
        values = []
        for sample_idx in range(len(self.complex_files) * self.multiplicity):
            file_idx = sample_idx % len(self.complex_files)
            candidate_idx = (sample_idx // len(self.complex_files))
            name = self.complex_files[file_idx].removeprefix("heterograph-").rsplit("-", 1)[0]
            overlap = float(overlap_by_complex[name][candidate_idx])
            if overlap < thresholds["p75"]:
                level = 0
            elif overlap < thresholds["p90"]:
                level = 1
            elif overlap < thresholds["p95"]:
                level = 2
            else:
                level = 3
            values.append(float(level_weights[level]))
        return torch.tensor(values, dtype=torch.double)

    def prepare_clustering_dicts(self) -> None:
        if self.cluster_file is not None:
            with open(self.cluster_file, "r", newline="") as file:
                reader = csv.DictReader(file)
                cluster_data = list(reader)

            self.all_complex_files = self.complex_files
            # Remove complexes that did not pass preprocessing and loading tests
            self.complex_to_cluster = {
                f"heterograph-{row['complex_name']}": row["cluster_id"]
                for row in cluster_data
                if f"heterograph-{row['complex_name']}" in self.complex_files
            }
            self.cluster_to_complex: dict[str, list] = {}
            for key, value in self.complex_to_cluster.items():
                if value not in self.cluster_to_complex:
                    self.cluster_to_complex[value] = [key]
                else:
                    self.cluster_to_complex[value].append(key)

    def subsample_clusters(self) -> None:
        if self.cluster_file is not None:
            subsampled_cluster_files = []
            for cluster_complexes in self.cluster_to_complex.values():
                random.shuffle(cluster_complexes)
                subsampled_cluster_files.extend(
                    cluster_complexes[: self.complexes_per_cluster]
                )
            self.complex_files = subsampled_cluster_files

    def check_processed_inputs(self):
        if not os.path.exists(self.cache_path):
            return False

        else:
            complex_names_all = read_strings_from_txt(self.split_path)
            if self.limit_complexes is not None and self.limit_complexes != 0:
                complex_names_all = complex_names_all[: self.limit_complexes]

            complexes_available = [
                complex_name
                for complex_name in complex_names_all
                if os.path.exists(f"{self.cache_path}/heterograph-{complex_name}-0.pt")
            ]

            # complexes_available = [
            #     filename.removeprefix("heterograph-").removesuffix("-0.pt")
            #     for filename in os.listdir(self.cache_path)
            #     if "heterograph" in filename
            # ]

            if not len(complexes_available):
                print("Directory found but no complexes available.", flush=True)
                return False

            if not len(set(complex_names_all).intersection(set(complexes_available))):
                print("No common complexes.", flush=True)
                return False

        return True

    def gather_processed_inputs(self):
        print(
            f"Loading each complex individually from: {self.cache_path}",
            flush=True,
        )
        print(flush=True)
        complex_names_all = read_strings_from_txt(self.split_path)

        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]

        self.complex_files = [
            f"heterograph-{name}-0"
            for name in complex_names_all
            if name not in self.exclude_complexes
            if os.path.exists(f"{self.cache_path}/heterograph-{name}-0.pt")
            if self.interaction_cache_root is None
            or os.path.exists(
                os.path.join(
                    self.interaction_cache_root, name, f"interaction_{name}_v2.pkl"
                )
            )
            if self.stage2_initial_pose_root is None
            or self.stage2_pose_manifest is not None
            or os.path.exists(
                os.path.join(
                    self.stage2_initial_pose_root,
                    self.stage2_split,
                    "predictions",
                    name,
                    "docking_predictions.pkl",
                )
            )
        ]
        manifest_path = self._manifest_path()
        if manifest_path is not None:
            with open(manifest_path) as handle:
                payload = json.load(handle)
            allowed = {
                name.removeprefix("heterograph-").rsplit("-", 1)[0]
                for name in self.complex_files
            }
            self.stage2_samples = [
                sample for sample in payload["samples"]
                if sample["complex_id"] in allowed
                and os.path.isfile(sample["pose_file"])
            ]
            if not self.stage2_samples:
                raise ValueError(f"No usable Stage-2 samples in {manifest_path}")
            print(
                f"Using {len(self.stage2_samples)} qualified Stage-2 poses from "
                f"{len(set(s['complex_id'] for s in self.stage2_samples))} complexes "
                f"({manifest_path})",
                flush=True,
            )
        if self.interaction_cache_root is not None:
            print(
                f"Using {len(self.complex_files)} complexes with EC-Dock interaction caches",
                flush=True,
            )

    def len(self):
        if hasattr(self, "stage2_samples"):
            return len(self.stage2_samples)
        return len(self.complex_files) * self.multiplicity

    def get(self, idx):
        sample_idx = idx
        manifest_sample = None
        if hasattr(self, "stage2_samples"):
            manifest_sample = self.stage2_samples[idx]
            name = f"heterograph-{manifest_sample['complex_id']}-0"
            idx = None
        elif self.multiplicity:
            idx = idx % len(self.complex_files)
            name = self.complex_files[idx]
        else:
            name = self.complex_files[idx]
        # Cached PyG HeteroData graphs are trusted local preprocessing outputs;
        # PyTorch >=2.6 otherwise defaults to weights_only=True and refuses
        # to unpickle their ComplexData type.
        if name in self._graph_memory_cache:
            # HeteroData.clone() does not guarantee isolation for every nested
            # Python/NumPy attribute used by our in-place coordinate transforms.
            complex_graph = copy.deepcopy(self._graph_memory_cache[name])
        else:
            complex_graph = torch.load(
                f"{self.cache_path}/{name}.pt", weights_only=False
            )
        if self.paired_distance_conformer:
            complex_name = name.removeprefix("heterograph-").rsplit("-", 1)[0]
            interaction_path = (
                os.path.join(
                    self.interaction_cache_root, complex_name,
                    f"interaction_{complex_name}_v2.pkl",
                )
                if self.interaction_cache_root is not None
                else os.path.join(
                    os.path.dirname(str(complex_graph.apo_rec_path)),
                    f"interaction_{complex_name}_v2.pkl",
                )
            )
            if complex_name in self._interaction_memory_cache:
                paired_payload = self._interaction_memory_cache[complex_name]
            else:
                with open(interaction_path, "rb") as handle:
                    paired_payload = pickle.load(handle)
            distances = paired_payload.get("cross_distance_list", [])
            raw = paired_payload.get("rdkit_candidate_coords_list", [])
            matched = paired_payload.get("rdkit_matched_candidate_coords_list", [])
            offsets = paired_payload.get("rdkit_matched_candidate_tor_offsets_list", [])
            matching_rmsds = paired_payload.get("rdkit_matched_candidate_rmsd_list", [])
            count = len(distances)
            if not (count > 0 and len(raw) == count):
                raise ValueError(
                    f"{complex_name}: incomplete paired distance/raw-RDKit cache"
                )
            # Cartesian flow only needs a distance matrix and its paired raw
            # RDKit conformer.  Older otherwise-valid caches may not contain
            # the torsion-matching fields used by the internal-coordinate
            # model.  Supply inert compatibility values instead of dropping
            # those complexes from Cartesian training.
            matching_available = (
                len(matched) == count and len(offsets) == count
                and len(matching_rmsds) == count
            )
            if not matching_available:
                matched = raw
                n_torsions = int(complex_graph["ligand"].edge_mask.sum())
                offsets = [np.zeros(n_torsions, dtype=np.float32) for _ in range(count)]
                matching_rmsds = [float("nan")] * count
            eligible = list(range(count))
            if self.paired_candidate_matching_rmsd_max is not None:
                if not matching_available:
                    raise ValueError(
                        f"{complex_name}: matching-RMSD filtering requested but "
                        "matching metadata is unavailable"
                    )
                threshold = float(self.paired_candidate_matching_rmsd_max)
                eligible = [
                    i for i, value in enumerate(matching_rmsds)
                    if float(value) <= threshold
                ]
                if not eligible:
                    raise ValueError(
                        f"{complex_name}: no paired candidate has matching RMSD "
                        f"<= {threshold:.3f} A"
                    )
            candidate = eligible[int(torch.randint(len(eligible), (1,)).item())]
            expected_shape = tuple(complex_graph["ligand"].pos.shape)
            # torch.as_tensor(np_array) aliases the cached NumPy allocation.
            # The flow transform mutates coordinates in place, so materialise
            # independent tensors for every sample.
            candidate_matched = torch.tensor(matched[candidate], dtype=torch.float32)
            candidate_raw = torch.tensor(raw[candidate], dtype=torch.float32)
            if tuple(candidate_raw.shape) != expected_shape:
                raise ValueError(
                    f"{complex_name}: paired ligand coordinate shape mismatch"
                )
            complex_graph["ligand"].pos = candidate_raw
            complex_graph["ligand"].rdkit_source_pos = candidate_raw
            if self.prepare_paired_torsion_metadata:
                candidate_offset = torch.tensor(offsets[candidate], dtype=torch.float32)
                if tuple(candidate_matched.shape) != expected_shape:
                    raise ValueError(
                        f"{complex_name}: matched ligand coordinate shape mismatch"
                    )
                if candidate_offset.numel() != int(complex_graph["ligand"].edge_mask.sum()):
                    raise ValueError(
                        f"{complex_name}: paired torsion offset count mismatch"
                    )
                complex_graph["ligand"].pos = candidate_matched
                complex_graph["ligand"].rdkit_source_tor_offset = candidate_offset
                canonical_mol = _restore_aligned_rdkit_mol(
                    complex_graph, self.interaction_cache_root, complex_name
                )
                complex_graph["ligand"].rdkit_source_tor_period = _chemical_torsion_periods(
                    complex_graph
                )
                attach_canonical_torsions(complex_graph, mol=canonical_mol)
                _attach_torsion_reliability(complex_graph)
            complex_graph.distance_guidance_candidate_index = candidate
            complex_graph.paired_candidate_matching_rmsd = float(matching_rmsds[candidate])
        # Preserve the cache/PDB atom order and absolute coordinates before
        # stochastic matching, centering and diffusion transforms.  Fixed
        # EC-Dock distance matrices are indexed in exactly this coordinate
        # frame and must never be mapped through the current noisy x_t.
        complex_graph["ligand"].ecdock_reference_pos = (
            complex_graph["ligand"].pos.detach().clone()
        )
        complex_graph["atom"].ecdock_reference_pos = (
            complex_graph["atom"].pos.detach().clone()
        )
        if self.stage2_initial_pose_root is not None:
            complex_name = name.removeprefix("heterograph-").rsplit("-", 1)[0]
            pose_file = (
                manifest_sample["pose_file"]
                if manifest_sample is not None
                else os.path.join(
                    self.stage2_initial_pose_root,
                    self.stage2_split,
                    "predictions",
                    complex_name,
                    "docking_predictions.pkl",
                )
            )
            with open(pose_file, "rb") as handle:
                poses = pickle.load(handle)["ligand_pos"]
            # Multiplicity deterministically exposes all candidates; shuffling
            # the training loader randomises their order between epochs.
            pose_idx = (
                int(manifest_sample["pose_index"])
                if manifest_sample is not None
                else (sample_idx // len(self.complex_files)) % len(poses)
            )
            pose = np.asarray(poses[pose_idx], dtype=np.float32)
            if pose.shape != tuple(complex_graph["ligand"].pos.shape):
                raise ValueError(
                    f"Stage-2 pose shape mismatch for {complex_name}: "
                    f"{pose.shape} != {tuple(complex_graph['ligand'].pos.shape)}"
                )
            complex_graph["ligand"].stage2_initial_pos = torch.from_numpy(pose)
        return complex_graph


class DockingDataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg: DockingDataConfig,
        transform_cfg: dict[str, Any],
    ):
        super().__init__()
        assert (
            data_cfg.dataset in AVAILABLE_DATASETS
        ), f"Dataset={data_cfg.dataset} not found in {AVAILABLE_DATASETS}"

        self.data_cfg = data_cfg
        self.transform_cfg = transform_cfg

        train_transform = construct_transform(cfg=transform_cfg, mode="train")
        self._train_dataset = DockingDataset(
            transform=train_transform,
            dataset=data_cfg.dataset,
            cache_path=data_cfg.cache_path,
            split_path=data_cfg.split_train,
            cluster_file=data_cfg.cluster_file,
            complexes_per_cluster=data_cfg.complexes_per_cluster,
            limit_complexes=data_cfg.limit_complexes,
            multiplicity=data_cfg.multiplicity,
            num_workers=data_cfg.num_workers,
            require_ligand=data_cfg.require_ligand,
            exclude_complexes=data_cfg.exclude_complexes,
            interaction_cache_root=data_cfg.interaction_cache_root,
            paired_distance_conformer=data_cfg.paired_distance_conformer,
            paired_candidate_matching_rmsd_max=data_cfg.paired_candidate_matching_rmsd_max,
            prepare_paired_torsion_metadata=data_cfg.prepare_paired_torsion_metadata,
            stage2_initial_pose_root=data_cfg.stage2_initial_pose_root,
            stage2_pose_manifest=data_cfg.stage2_pose_manifest,
            stage2_split="train",
            hard_overlap_cache=data_cfg.hard_overlap_cache,
            preload_to_memory=data_cfg.preload_to_memory,
        )

        self._val_dataset = None
        if not data_cfg.disable_validation:
            val_transform = construct_transform(cfg=transform_cfg, mode="val")
            self._val_dataset = DockingDataset(
                transform=val_transform,
                dataset=data_cfg.dataset,
                cache_path=data_cfg.cache_path,
                split_path=data_cfg.split_val,
                cluster_file=data_cfg.cluster_file,
                complexes_per_cluster=data_cfg.complexes_per_cluster,
                limit_complexes=data_cfg.limit_complexes,
                multiplicity=min(data_cfg.multiplicity, 5),
                num_workers=data_cfg.num_workers,
                require_ligand=data_cfg.require_ligand,
                exclude_complexes=data_cfg.exclude_complexes,
                interaction_cache_root=data_cfg.interaction_cache_root,
                paired_distance_conformer=data_cfg.paired_distance_conformer,
                paired_candidate_matching_rmsd_max=data_cfg.paired_candidate_matching_rmsd_max,
                prepare_paired_torsion_metadata=data_cfg.prepare_paired_torsion_metadata,
                stage2_initial_pose_root=data_cfg.stage2_initial_pose_root,
                stage2_pose_manifest=data_cfg.stage2_pose_manifest,
                stage2_split="val",
                preload_to_memory=data_cfg.preload_to_memory,
            )

        if data_cfg.run_val_inference:
            if self._val_dataset is None:
                raise ValueError("run_val_inference is incompatible with disable_validation")
            inf_transform = construct_transform(cfg=transform_cfg, mode="inference")
            inf_complexes = [
                self._val_dataset.get(idx)
                for idx in range(
                    min(data_cfg.num_inference_complexes, len(self._val_dataset))
                )
            ]
            if len(inf_complexes) == 1:
                inf_complexes = inf_complexes * 20

            self._inf_dataset = ListDataset(inf_complexes, transform=inf_transform)

    def setup(self, stage):
        return

    def train_dataloader(self):
        sampler = None
        if self.data_cfg.hard_overlap_sampling:
            if self._train_dataset.sampling_weights is None:
                raise ValueError("hard_overlap_sampling requires hard_overlap_cache")
            sampler = DistributedWeightedSampler(
                self._train_dataset.sampling_weights, seed=42
            )
        train_loader = DataLoader(
            dataset=self._train_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_dataloader_workers,
            shuffle=sampler is None,
            sampler=sampler,
            pin_memory=False,
            drop_last=self.data_cfg.dataloader_drop_last,
        )
        return train_loader

    def val_dataloader(self):
        if self._val_dataset is None:
            return None
        val_loader = DataLoader(
            dataset=self._val_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.num_dataloader_workers,
            shuffle=False,
            pin_memory=self.data_cfg.pin_memory,
            drop_last=self.data_cfg.dataloader_drop_last,
        )
        val_loaders = [val_loader]

        if self.data_cfg.run_val_inference:
            inf_dataset = self._inf_dataset
            val_loaders.append(
                DataLoader(dataset=inf_dataset, batch_size=1, shuffle=False)
            )

        loader = CombinedLoader(val_loaders, mode="sequential")
        return loader
