#!/usr/bin/env python3
"""Export IFMScore-ranked poses and crystal structures by complex."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from rdkit import Chem


def export_one(name: str, epoch_dir: Path, data_dir: Path, output_dir: Path) -> int:
    complex_dir = output_dir / name
    complex_dir.mkdir(parents=True, exist_ok=True)
    record_path = epoch_dir / "ifmscore" / f"{name}.json"
    record = json.loads(record_path.read_text())
    if record.get("complex_id") != name:
        raise ValueError(f"IFMScore record does not match {name}")
    if record.get("status") != "ok":
        raise ValueError(f"IFMScore failed for {name}: {record.get('reason', 'unknown reason')}")

    indices = [int(value) for value in record["sample_indices"]]
    scores = [float(value) for value in record["scores"]]
    if (not indices or len(indices) != len(scores) or len(set(indices)) != len(indices)
            or any(not math.isfinite(score) for score in scores)):
        raise ValueError(f"Invalid IFMScore indices or scores for {name}")

    source = epoch_dir / "predictions" / name / f"gen_{name}_ligand.sdf"
    poses = [mol for mol in Chem.SDMolSupplier(str(source), removeHs=False) if mol]
    if any(index < 0 or index >= len(poses) for index in indices):
        raise IndexError(f"IFMScore pose index outside the SDF for {name}")

    crystal_dir = data_dir / name
    crystal_ligand = crystal_dir / f"{name}_ligand.sdf"
    protein = crystal_dir / f"origin_{name}_protein.pdb"
    if not protein.is_file():
        protein = crystal_dir / f"{name}_protein.pdb"
    for path in (crystal_ligand, protein):
        if not path.is_file():
            raise FileNotFoundError(path)

    ranked = sorted(zip(indices, scores), key=lambda item: (-item[1], item[0]))
    fd, temporary = tempfile.mkstemp(prefix=f".gen_{name}_", suffix=".sdf", dir=complex_dir)
    os.close(fd)
    try:
        writer = Chem.SDWriter(temporary)
        try:
            for rank, (index, score) in enumerate(ranked, 1):
                pose = Chem.Mol(poses[index])
                pose.SetIntProp("IFMScore_rank", rank)
                pose.SetDoubleProp("IFMScore", score)
                pose.SetIntProp("source_pose_index", index)
                writer.write(pose)
        finally:
            writer.close()
        os.chmod(temporary, 0o644)
        os.replace(temporary, complex_dir / f"gen_{name}_ligand.sdf")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    shutil.copy2(crystal_ligand, complex_dir / f"{name}_ligand.sdf")
    shutil.copy2(protein, complex_dir / f"{name}_protein.pdb")
    return len(ranked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.epoch_dir / "inputs.csv").open(newline="") as handle:
        names = [row["pdbid"] for row in csv.DictReader(handle)]
    for name in names:
        count = export_one(name, args.epoch_dir, args.data_dir, args.output_dir)
        print(f"{name}: exported {count} ranked poses and crystal structures", flush=True)


if __name__ == "__main__":
    main()
