#!/usr/bin/env python3
"""Persist EC-Dock-to-PyG index permutations for inference datasets.

Coordinates are used once here to prove the protein permutation. Runtime edge
construction then consumes only the stored integer arrays.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ifmdock.data.feature.featurizer import FeaturizerConfig
from ifmdock.data.modules.inference import PredictionDataset
from ifmdock.preprocessing.atom_identity import ligand_atom_ids, pyg_protein_atom_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()
    frame = pd.read_csv(args.inputs)
    cfg = FeaturizerConfig(
        matching=False, popsize=None, maxiter=None, keep_original=False,
        remove_hs=True, num_conformers=1, max_lig_size=None,
        flexible_backbone=False, flexible_sidechains=False, rigid_pocket=True,
    )
    dataset = PredictionDataset(str(args.inputs), cfg, rigid_pocket=True)
    ok, failures = 0, []
    for index, row in frame.iterrows():
        name = str(row.pdbid)
        path = Path(row.base_dir) / f"interaction_{name}_v2.pkl"
        try:
            graph = dataset.get(index)
            graph_coords = np.asarray(graph["atom"].orig_apo_pos, dtype=np.float64)
            ligand_count = int(graph["ligand"].pos.shape[0])
            protein_ids = pyg_protein_atom_ids(row.apo_protein_file)
            ligand_ids = ligand_atom_ids(row.ligand_path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            pockets, distances, max_error = [], [], 0.0
            for candidate, (pocket, distance) in enumerate(zip(
                payload["pocket_coords_list"], payload["cross_distance_list"]
            )):
                pocket = np.asarray(pocket, dtype=np.float64)
                distance = np.asarray(distance)
                if pocket.shape != graph_coords.shape:
                    raise ValueError(f"candidate {candidate}: protein shape mismatch")
                if distance.shape[0] != ligand_count:
                    raise ValueError(f"candidate {candidate}: ligand count mismatch")
                # Caches entering this migration were already standardized.
                # Establish atom identity from molecular/PDB records; use the
                # coordinates only to audit that the stored order agrees.
                errors = np.linalg.norm(pocket - graph_coords, axis=1)
                if errors.max(initial=0.0) > args.tolerance:
                    raise ValueError(f"protein identity audit error {errors.max():.6f} A")
                pockets.append(np.asarray(pocket, dtype=np.float32))
                distances.append(distance)
                max_error = max(max_error, float(errors.max(initial=0.0)))
            payload["pocket_coords_list"] = pockets
            payload["cross_distance_list"] = distances
            payload["ifmdock_atom_order"] = {
                "version": 1, "atom_count": len(graph_coords),
                "max_mapping_error": max_error,
            }
            payload["ifmdock_index_mapping"] = {
                "version": 1,
                "method": "chemical_identity_v1",
                "ligand_to_pyg": np.arange(ligand_count, dtype=np.int64),
                "protein_to_pyg": np.arange(len(graph_coords), dtype=np.int64),
                "runtime_coordinate_mapping": False,
                "coordinates_used_for_mapping": False,
                "coordinate_audit_max_error": max_error,
            }
            if len(ligand_ids) != ligand_count or len(protein_ids) != len(graph_coords):
                raise ValueError("chemical identity count mismatch")
            payload["distance_ligand_atom_ids"] = ligand_ids
            payload["pyg_ligand_atom_ids"] = ligand_ids
            payload["distance_protein_atom_ids"] = protein_ids
            payload["pyg_protein_atom_ids"] = protein_ids
            temporary = path.with_suffix(path.suffix + ".index.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
            ok += 1
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    print(f"standardized={ok} failed={len(failures)}")
    for item in failures[:20]:
        print(*item, sep="\t")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
