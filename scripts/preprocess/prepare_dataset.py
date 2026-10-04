#!/usr/bin/env python3
"""Prepare cleaned receptors, ligands, docking boxes and UniMol256 pockets.

The raw receptor is preserved as ``origin_<id>_protein.pdb``. The working
``<id>_protein.pdb`` is atomically replaced by an EC-Dock-compatible receptor
containing only non-hydrogen atoms from the 20 standard amino acids. Reference
ligand coordinates are preserved and every generated artifact is written
atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from ifmdock.preprocessing.unimol_pocket import (
    clean_full_protein,
    generate_unimol_pocket,
    read_docking_grid,
    write_unimol_pocket,
)


def fix_quaternary_nitrogen(mol: Chem.Mol) -> Chem.Mol:
    editable = Chem.RWMol(mol)
    editable.UpdatePropertyCache(strict=False)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 0:
            continue
        try:
            if atom.GetExplicitValence() == 4:
                atom.SetFormalCharge(1)
        except RuntimeError:
            pass
    result = editable.GetMol()
    result.UpdatePropertyCache(strict=False)
    return result


def read_ligand(directory: Path, name: str) -> tuple[Chem.Mol, str]:
    sdf = directory / f"{name}_ligand.sdf"
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)
    mol = supplier[0] if len(supplier) else None
    if mol is not None and mol.GetNumConformers():
        return mol, "strict_sdf"

    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
    mol = supplier[0] if len(supplier) else None
    if mol is not None:
        mol = fix_quaternary_nitrogen(Chem.Mol(mol))
        try:
            Chem.SanitizeMol(mol)
            if mol.GetNumConformers():
                return mol, "repaired_sdf"
        except Exception:
            pass

    mol2 = directory / f"{name}_ligand.mol2"
    if mol2.is_file():
        mol = Chem.MolFromMol2File(str(mol2), removeHs=False, sanitize=False)
        if mol is not None:
            mol = fix_quaternary_nitrogen(mol)
            Chem.SanitizeMol(mol)
            if mol.GetNumConformers():
                return mol, "repaired_mol2"
    raise ValueError("RDKit could not obtain a sanitized ligand with 3D coordinates")


def atomic_sdf(mol: Chem.Mol, target: Path, backup: bool) -> None:
    if backup:
        original = target.with_name(f"origin_{target.name}")
        if not original.exists():
            shutil.copy2(target, original)
    temporary = target.with_suffix(".sdf.ifmdock.tmp")
    writer = Chem.SDWriter(str(temporary))
    writer.write(mol)
    writer.close()
    check = Chem.SDMolSupplier(str(temporary), removeHs=False, sanitize=True)
    if not len(check) or check[0] is None or not check[0].GetNumConformers():
        temporary.unlink(missing_ok=True)
        raise ValueError("repaired ligand failed write-back validation")
    os.replace(temporary, target)


def clean_and_backup_protein(protein: Path, name: str) -> tuple[int, int]:
    """Back up the raw receptor and overwrite it with EC-Dock-style cleaning."""
    backup = protein.with_name(f"origin_{name}_protein.pdb")
    if not backup.is_file():
        shutil.copy2(protein, backup)
    cleaned, metadata = clean_full_protein(backup)
    write_unimol_pocket(cleaned, protein)
    return metadata["input_atom_records"], metadata["selected_atoms"]


def ligand_heavy_coordinates(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=np.float64)
    mask = np.array([atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()], dtype=bool)
    coords = coords[mask]
    if not len(coords) or not np.isfinite(coords).all():
        raise ValueError("ligand has no finite heavy-atom coordinates")
    return coords


def make_grid(coords: np.ndarray, padding: float) -> dict[str, float]:
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0) + float(padding)
    return {
        **{f"center_{axis}": float(center[i]) for i, axis in enumerate("xyz")},
        **{f"size_{axis}": float(extent[i]) for i, axis in enumerate("xyz")},
    }


def process_one(task):
    directory, padding, max_atoms, overwrite, repair_ligands, backup = task
    directory = Path(directory)
    name = directory.name
    row = {"complex": name, "status": "failed", "ligand_source": "", "box_atoms": 0,
           "selected_atoms": 0, "grid_written": False, "pocket_written": False,
           "protein_filtered": False, "removed_residues": "", "detail": ""}
    try:
        protein = directory / f"{name}_protein.pdb"
        ligand_path = directory / f"{name}_ligand.sdf"
        if not protein.is_file() or not ligand_path.is_file():
            raise FileNotFoundError("missing protein.pdb or ligand.sdf")
        input_atoms, cleaned_atoms = clean_and_backup_protein(protein, name)
        row["protein_filtered"] = cleaned_atoms != input_atoms
        row["removed_residues"] = f"removed_atom_records={input_atoms-cleaned_atoms}"
        ligand, source = read_ligand(directory, name)
        row["ligand_source"] = source
        if source != "strict_sdf" and repair_ligands:
            atomic_sdf(ligand, ligand_path, backup)

        grid = make_grid(ligand_heavy_coordinates(ligand), padding)
        grid_path = directory / f"{name}_ligand_docking_grid_boxsize10.json"
        if overwrite or not grid_path.is_file():
            temporary = grid_path.with_suffix(".json.ifmdock.tmp")
            temporary.write_text(json.dumps(grid, indent=2) + "\n")
            os.replace(temporary, grid_path)
            row["grid_written"] = True
        else:
            grid = read_docking_grid(grid_path)

        selected, pocket_info = generate_unimol_pocket(protein, grid, max_atoms)
        pocket = directory / f"{name}_protein_{max_atoms}.pdb"
        if overwrite or not pocket.is_file():
            write_unimol_pocket(selected, pocket)
            row["pocket_written"] = True
        row.update(status="ok", box_atoms=pocket_info["box_atoms"], selected_atoms=len(selected))
    except Exception as exc:
        row["detail"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--padding", type=float, default=10.0)
    parser.add_argument("--max-pocket-atoms", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Atomically replace existing grid and pocket outputs.")
    parser.add_argument("--repair-ligands", action="store_true",
                        help="Overwrite only ligands that fail strict RDKit parsing.")
    parser.add_argument("--backup-repaired-ligands", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.max_pocket_atoms < 1 or args.padding <= 0 or args.workers < 1:
        parser.error("workers, padding and max-pocket-atoms must be positive")
    directories = sorted(path for path in args.dataset.iterdir() if path.is_dir())
    if args.limit:
        directories = directories[:args.limit]
    tasks = ((str(path), args.padding, args.max_pocket_atoms, args.overwrite,
              args.repair_ligands, args.backup_repaired_ligands) for path in directories)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(process_one, tasks, chunksize=8), 1):
            rows.append(row)
            if index % 500 == 0:
                print(f"processed={index}/{len(directories)} failed={sum(r['status'] != 'ok' for r in rows)}",
                      flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fields = ["complex", "status", "ligand_source", "box_atoms", "selected_atoms",
              "grid_written", "pocket_written", "protein_filtered",
              "removed_residues", "detail"]
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, args.report)
    failed = sum(row["status"] != "ok" for row in rows)
    repaired = sum(row["ligand_source"].startswith("repaired") for row in rows)
    print(json.dumps({"total": len(rows), "ok": len(rows) - failed, "failed": failed,
                      "repair_candidates": repaired, "report": str(args.report)}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
