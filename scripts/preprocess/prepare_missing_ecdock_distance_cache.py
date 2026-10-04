#!/usr/bin/env python3
"""Prepare/finalize sharded EC-Dock distance-cache generation."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np

from ifmdock import get_atom_order_metadata


def ready_cache(path: Path):
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        info = get_atom_order_metadata(payload)
        count = len(payload.get("cross_distance_list", []))
        return (
            isinstance(info, dict) and info.get("version") in (0, 1)
            and count > 0
            and len(payload.get("pocket_coords_list", [])) == count
        )
    except Exception:
        return False


def missing_names(
    split_dir: Path, dataset: Path, graph_cache: Path | None = None,
    exclude=(),
):
    excluded = set(exclude)
    names = []
    for split in ("train.txt", "val.txt"):
        for name in (x.strip() for x in (split_dir / split).read_text().splitlines()):
            graph_available = (
                graph_cache is None
                or (graph_cache / f"heterograph-{name}-0.pt").is_file()
            )
            if (
                name and name not in excluded and graph_available
                and not ready_cache(dataset / name / f"interaction_{name}_v2.pkl")
            ):
                names.append(name)
    return sorted(set(names))


def split_names(split_dir: Path):
    names = []
    for split in ("train.txt", "val.txt"):
        names.extend(x.strip() for x in (split_dir / split).read_text().splitlines())
    return sorted(set(x for x in names if x))


def prepare(args):
    graph_cache = None if args.skip_graph_standardization else args.graph_cache
    names = (
        [n for n in split_names(args.split_dir) if n not in set(args.exclude)]
        if args.force else
        missing_names(args.split_dir, args.dataset, graph_cache, args.exclude)
    )
    # Restartability: an output may already have been generated but not yet
    # installed into the dataset.  Do not schedule it again when increasing or
    # otherwise changing the shard count.
    pending = []
    reused_outputs = 0
    for name in names:
        output = args.work_dir / "outputs" / name / f"interaction_{name}.pkl"
        ok, _ = valid_payload(output, args.conf_size) if output.is_file() else (False, "missing")
        if ok:
            reused_outputs += 1
        else:
            pending.append(name)
    names = pending
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # A previous run may have used more shards.  Remove stale manifests before
    # writing the current set so the finalizer cannot count obsolete work.
    for stale_manifest in args.work_dir.glob("missing_shard_*.csv"):
        stale_manifest.unlink()
    fieldnames = [
        "input_protein", "input_ligand", "input_docking_grid",
        "output_ligand_name", "output_ligand_dir2",
    ]
    valid, invalid = [], []
    for name in names:
        source = args.dataset / name
        row = {
            "input_protein": source / f"{name}_protein_256.pdb",
            "input_ligand": (
                args.ligand_override_root / f"{name}_ligand.sdf"
                if args.ligand_override_root is not None
                and (args.ligand_override_root / f"{name}_ligand.sdf").is_file()
                else source / f"{name}_ligand.sdf"
            ),
            "input_docking_grid": source / f"{name}_ligand_docking_grid_boxsize10.json",
            "output_ligand_name": name,
            "output_ligand_dir2": args.work_dir / "outputs" / name,
        }
        absent = [str(path) for key, path in row.items() if key.startswith("input_") and not path.is_file()]
        if absent:
            invalid.append((name, ";".join(absent)))
        else:
            valid.append({key: str(value) for key, value in row.items()})
    for shard in range(args.shards):
        path = args.work_dir / f"missing_shard_{shard}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid[shard::args.shards])
    with (args.work_dir / "invalid_inputs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["complex", "missing_files"])
        writer.writerows(invalid)
    print(
        f"pending={len(names)} reused_outputs={reused_outputs} "
        f"runnable={len(valid)} invalid={len(invalid)} shards={args.shards}"
    )


def valid_payload(path: Path, conf_size: int):
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        keys = ("holo_coords_list", "coords_predict_list", "pocket_coords_list", "cross_distance_list")
        lengths = [len(payload[key]) for key in keys]
        if min(lengths) <= 0 or len(set(lengths)) != 1:
            return False, f"invalid lengths {lengths}"
        # Some molecules can lose a failed TTA conformer; retain them as long
        # as at least one fully aligned candidate survived postprocessing.
        for h, p, d in zip(
            payload["holo_coords_list"], payload["pocket_coords_list"],
            payload["cross_distance_list"],
        ):
            if tuple(d.shape) != (len(h), len(p)):
                return False, f"shape mismatch {d.shape} != {(len(h), len(p))}"
        return True, f"candidates={lengths[0]}/{conf_size}"
    except Exception as error:
        return False, repr(error)


def standardize_protein_order(payload, graph_path: Path, tolerance: float, path_root: Path | None = None):
    """Put every EC-Dock pocket column in the cached IFMDock atom order.

    EC-Dock preserves PDB row order, whereas IFMDock sorts atoms inside each
    residue with ``ATOM_ORDER_DICT``.  Both therefore contain the same atoms
    but cannot safely share integer indices.  Resolve the permutation once,
    offline, and store distances in the graph's native order.  Runtime code can
    subsequently use identity indices and only has to verify the coordinates.
    """
    import torch
    from ifmdock.preprocessing.atom_identity import pyg_protein_atom_ids

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_coords = np.asarray(graph["atom"].orig_apo_pos, dtype=np.float64)
    apo_path = graph["apo_rec_path"]
    while isinstance(apo_path, (list, tuple)):
        apo_path = apo_path[0]
    # Graphs assembled from a relative dataset path retain that relative path.
    # Finalization is commonly launched from the IFMDock repository rather than
    # the dataset's owner repository, so resolve it deterministically against
    # the supplied dataset root before asking BioPandas to read it.
    apo_path = Path(apo_path)
    if not apo_path.is_file() and path_root is not None:
        candidate = path_root / apo_path
        if candidate.is_file():
            apo_path = candidate
    protein_ids = pyg_protein_atom_ids(apo_path)
    if len(protein_ids) != len(graph_coords):
        raise ValueError("protein identity/graph atom count mismatch")
    candidate_count = len(payload.get("cross_distance_list", []))
    if candidate_count == 0:
        raise ValueError("empty cross_distance_list")
    if len(payload.get("pocket_coords_list", [])) != candidate_count:
        raise ValueError("unaligned pocket_coords_list and cross_distance_list")
    standardized_pockets = []
    standardized_distances = []
    max_errors = []
    for candidate, (pocket, distance) in enumerate(zip(
        payload["pocket_coords_list"], payload["cross_distance_list"]
    )):
        pocket = np.asarray(pocket, dtype=np.float64)
        distance = np.asarray(distance)
        if pocket.shape != graph_coords.shape:
            raise ValueError(
                f"candidate {candidate}: pocket shape {pocket.shape} != "
                f"IFMDock atom shape {graph_coords.shape}"
            )
        # New EC-Dock caches must persist their source identities.  Use only
        # those identities to establish the permutation; coordinates below
        # are an audit assertion and never select an atom.
        source_ids = payload.get("distance_protein_atom_ids")
        if source_ids is None:
            raise ValueError("raw cache lacks distance_protein_atom_ids")
        from ifmdock.preprocessing.atom_identity import identity_permutation
        source_to_pyg = np.asarray(identity_permutation(source_ids, protein_ids))
        pyg_to_source = np.argsort(source_to_pyg)
        reordered = np.asarray(pocket[pyg_to_source], dtype=np.float32)
        errors = np.linalg.norm(reordered.astype(np.float64) - graph_coords, axis=1)
        if errors.max(initial=0.0) > tolerance:
            raise ValueError(f"candidate {candidate}: identity audit error {errors.max():.6f} A")
        standardized_pockets.append(reordered)
        standardized_distances.append(distance[:, pyg_to_source])
        max_errors.append(float(errors.max(initial=0.0)))

    payload["pocket_coords_list"] = standardized_pockets
    payload["cross_distance_list"] = standardized_distances
    payload["ifmdock_atom_order"] = {
        "version": 1,
        "graph_file": graph_path.name,
        "atom_count": int(len(graph_coords)),
        "max_mapping_error": max(max_errors, default=0.0),
    }
    payload["distance_protein_atom_ids"] = protein_ids
    payload["pyg_protein_atom_ids"] = protein_ids
    ligand_count = int(len(np.asarray(payload["holo_coords_list"][0])))
    payload["ifmdock_index_mapping"] = {
        "version": 1,
        "method": "chemical_identity_v1",
        "ligand_to_pyg": np.arange(ligand_count, dtype=np.int64),
        # Protein columns were reordered above into native PyG order.
        "protein_to_pyg": np.arange(len(graph_coords), dtype=np.int64),
        "runtime_coordinate_mapping": False,
    }
    return payload


def finalize(args):
    graph_cache = None if args.skip_graph_standardization else args.graph_cache
    names = (
        [n for n in split_names(args.split_dir) if n not in set(args.exclude)]
        if args.force else
        missing_names(args.split_dir, args.dataset, graph_cache, args.exclude)
    )
    installed, failed = 0, []
    for name in names:
        source = args.work_dir / "outputs" / name / f"interaction_{name}.pkl"
        ok, detail = valid_payload(source, args.conf_size) if source.is_file() else (False, "missing output")
        if not ok:
            failed.append((name, detail))
            continue
        try:
            with source.open("rb") as handle:
                payload = pickle.load(handle)
            if args.skip_graph_standardization:
                # PoseBusters graphs are built at inference time.  Runtime
                # performs the exact coordinate bijection once per complex.
                payload["ifmdock_atom_order"] = {
                    "version": 0,
                    "runtime_coordinate_mapping": True,
                    "atom_count": int(len(payload["pocket_coords_list"][0])),
                }
            if args.rdkit_coords_root is not None:
                coords_file = args.rdkit_coords_root / f"{name}.npy"
                if coords_file.is_file():
                    coords = np.asarray(np.load(coords_file), dtype=np.float32)
                    expected = np.asarray(payload["holo_coords_list"][0]).shape
                    if coords.shape != expected:
                        raise ValueError(f"{name}: RDKit coordinate shape {coords.shape} != cache {expected}")
                    payload["rdkit_initial_coords"] = coords
            else:
                graph_path = args.graph_cache / f"heterograph-{name}-0.pt"
                if not graph_path.is_file():
                    raise FileNotFoundError(f"missing IFMDock graph: {graph_path}")
                import torch
                from ifmdock.preprocessing.atom_identity import (
                    distance_protein_atom_ids, ligand_atom_ids,
                )
                graph = torch.load(graph_path, map_location="cpu", weights_only=False)
                apo_path = graph["apo_rec_path"]
                while isinstance(apo_path, (list, tuple)):
                    apo_path = apo_path[0]
                apo_path = Path(apo_path)
                if not apo_path.is_file():
                    candidate = args.dataset.parents[1] / apo_path
                    if candidate.is_file():
                        apo_path = candidate
                payload["distance_protein_atom_ids"] = distance_protein_atom_ids(apo_path)
                ligand_files = sorted((args.dataset / name).glob("*_ligand.sdf"))
                if not ligand_files:
                    raise FileNotFoundError(f"{name}: source ligand SDF not found")
                ligand_ids = ligand_atom_ids(ligand_files[0])
                payload["distance_ligand_atom_ids"] = ligand_ids
                payload["pyg_ligand_atom_ids"] = ligand_ids
                payload = standardize_protein_order(
                    payload, graph_path, args.mapping_tolerance, args.dataset.parents[1]
                )
        except Exception as error:
            failed.append((name, f"atom-order standardization failed: {error!r}"))
            continue
        target = args.dataset / name / f"interaction_{name}_v2.pkl"
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
        installed += 1
    with (args.work_dir / "failed_outputs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["complex", "reason"])
        writer.writerows(failed)
    remaining = missing_names(args.split_dir, args.dataset, graph_cache, args.exclude)
    print(f"installed={installed} failed={len(failed)} remaining={len(remaining)}")


def standardize_installed(args):
    """Atomically migrate installed legacy PKLs to IFMDock atom ordering."""
    migrated, already, failed = 0, 0, []
    for name in split_names(args.split_dir):
        target = args.dataset / name / f"interaction_{name}_v2.pkl"
        graph_path = args.graph_cache / f"heterograph-{name}-0.pt"
        if not target.is_file() or not graph_path.is_file():
            failed.append((name, "missing interaction cache or IFMDock graph"))
            continue
        try:
            with target.open("rb") as handle:
                payload = pickle.load(handle)
            # Legacy inference caches predate persisted atom identities.  The
            # distance model read this exact UniMol256 PDB and source SDF, so
            # recover their stable chemical identities from those files once;
            # coordinates remain an audit only and never establish the map.
            from ifmdock.preprocessing.atom_identity import (
                distance_protein_atom_ids, ligand_atom_ids,
            )
            source_protein = args.dataset / name / f"{name}_protein_256.pdb"
            source_ligand = args.dataset / name / f"{name}_ligand.sdf"
            if payload.get("distance_protein_atom_ids") is None:
                payload["distance_protein_atom_ids"] = distance_protein_atom_ids(
                    source_protein
                )
            if payload.get("distance_ligand_atom_ids") is None:
                payload["distance_ligand_atom_ids"] = ligand_atom_ids(source_ligand)
            payload.setdefault("pyg_ligand_atom_ids", payload["distance_ligand_atom_ids"])
            info = get_atom_order_metadata(payload)
            has_candidates = len(payload.get("cross_distance_list", [])) > 0
            aligned_lists = (
                len(payload.get("pocket_coords_list", []))
                == len(payload.get("cross_distance_list", []))
            )
            if (
                isinstance(info, dict) and info.get("version") == 1
                and has_candidates and aligned_lists
            ):
                mapping = payload.get("ifmdock_index_mapping")
                if isinstance(mapping, dict) and mapping.get("version") == 1:
                    already += 1
                    continue
                # Historical caches may already have protein columns in the
                # graph order but lack the explicit ligand/protein permutation.
                # Ligand conformer matching changes coordinates, never RDKit
                # atom numbering, so identity is the correct ligand mapping.
                ligand_count = len(np.asarray(payload["holo_coords_list"][0]))
                protein_count = len(np.asarray(payload["pocket_coords_list"][0]))
                payload["ifmdock_index_mapping"] = {
                    "version": 1,
                    "method": "chemical_identity_v1",
                    "ligand_to_pyg": np.arange(ligand_count, dtype=np.int64),
                    "protein_to_pyg": np.arange(protein_count, dtype=np.int64),
                    "runtime_coordinate_mapping": False,
                }
                payload["ifmdock_atom_order"] = dict(info)
                temporary = target.with_suffix(target.suffix + ".order.tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, target)
                migrated += 1
                continue
            # Legacy inference caches (atom-order version 0) were deliberately
            # left in UniMol pocket order for runtime coordinate matching.
            # Current inference forbids that fragile mapping, so reorder their
            # protein columns once against the freshly constructed PyG graph
            # and persist an explicit chemical-identity permutation.
            payload = standardize_protein_order(
                payload, graph_path, args.mapping_tolerance, args.dataset.parents[1]
            )
            temporary = target.with_suffix(target.suffix + ".order.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, target)
            migrated += 1
        except Exception as error:
            failed.append((name, repr(error)))
    report = args.work_dir / "atom_order_migration_failures.csv"
    with report.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["complex", "reason"])
        writer.writerows(failed)
    print(f"atom_order migrated={migrated} already={already} failed={len(failed)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "finalize", "standardize"])
    parser.add_argument("--dataset", type=Path, default=Path("data/pdbbind"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/ecdock_distance_cache_completion"))
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--conf-size", type=int, default=10)
    parser.add_argument(
        "--graph-cache", type=Path,
        default=Path("data/cache"),
    )
    parser.add_argument("--mapping-tolerance", type=float, default=0.02)
    parser.add_argument("--force", action="store_true", help="Regenerate/install even when a ready cache exists.")
    parser.add_argument(
        "--ligand-override-root", type=Path, default=None,
        help="Use <root>/<complex>_ligand.sdf as the distance-model ligand input.",
    )
    parser.add_argument(
        "--rdkit-coords-root", type=Path, default=None,
        help="Store <root>/<complex>.npy in each installed PKL as rdkit_initial_coords.",
    )
    parser.add_argument(
        "--skip-graph-standardization", action="store_true",
        help="Install raw pocket order and defer coordinate mapping to inference.",
    )
    parser.add_argument("--exclude", nargs="*", default=["2r1w"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        standardize_installed(args)


if __name__ == "__main__":
    main()
