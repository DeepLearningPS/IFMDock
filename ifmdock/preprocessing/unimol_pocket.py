"""Single authoritative UniMol/EC-Dock atom-wise pocket implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from biopandas.pdb import PandasPdb


# These are the exact atom-name exclusions used by the validated UniMol/EC-Dock
# Processor path. They intentionally operate on PDB atom names, not elements.
DISALLOWED_TWO = {
    "Cd", "Cs", "Cn", "Ce", "Cm", "Cf", "Cl", "Ca", "Cr", "Co", "Cu",
    "Nh", "Nd", "Np", "No", "Ne", "Na", "Ni", "Nb", "Os", "Og", "Hf",
    "Hg", "Hs", "Ho", "He", "Sr", "Sn", "Sb", "Sg", "Sm", "Si", "Sc", "Se",
}
DISALLOWED_ONE = {"Z", "M", "P", "D", "F", "K", "I", "B"}
ATOM_IDENTITY = ["chain_id", "residue_number", "insertion", "atom_name"]
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def accepted_atom_name(value: str) -> bool:
    name = str(value).strip()
    while name and name[0].isdigit():
        name = name[1:]
    return bool(name) and name[:2] not in DISALLOWED_TWO and name[0] not in DISALLOWED_ONE


def read_docking_grid(path: str | Path) -> dict[str, float]:
    grid = json.loads(Path(path).read_text())
    required = {f"{kind}_{axis}" for kind in ("center", "size") for axis in "xyz"}
    missing = required.difference(grid)
    if missing:
        raise ValueError(f"docking grid is missing fields: {sorted(missing)}")
    result = {key: float(grid[key]) for key in required}
    if any(result[f"size_{axis}"] <= 0 for axis in "xyz"):
        raise ValueError("docking grid sizes must be positive")
    return result


def atom_signature(frame):
    """Identity, 0.001-A coordinates and order used for exact regression tests."""
    return list(zip(
        frame["chain_id"].astype(str), frame["residue_number"].astype(str),
        frame["insertion"].astype(str), frame["atom_name"].astype(str),
        frame["x_coord"].round(3), frame["y_coord"].round(3), frame["z_coord"].round(3),
    ))


def generate_unimol_pocket(
    protein_path: str | Path,
    grid: dict[str, float] | str | Path,
    max_atoms: int = 256,
):
    """Return the canonical strict-box, centroid-Top-K protein atom DataFrame.

    This is atom-wise cropping. Residues are deliberately not completed because
    the frozen UniMol distance model was trained with this pocket distribution.
    """
    if max_atoms < 1:
        raise ValueError("max_atoms must be positive")
    if not isinstance(grid, dict):
        grid = read_docking_grid(grid)
    else:
        grid = {key: float(value) for key, value in grid.items()}

    atoms = PandasPdb().read_pdb(str(protein_path)).df["ATOM"].copy()
    if atoms.empty:
        raise ValueError("protein contains no ATOM records")
    atoms = atoms[atoms["element_symbol"].astype(str).str.upper() != "H"].copy()
    atoms = atoms.drop_duplicates(ATOM_IDENTITY, keep="first")
    xyz = atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(np.float64)
    lower = np.array([grid[f"center_{axis}"] - grid[f"size_{axis}"] / 2 for axis in "xyz"])
    upper = np.array([grid[f"center_{axis}"] + grid[f"size_{axis}"] / 2 for axis in "xyz"])
    # UniMol/EC-Dock uses strict inequalities, so boundary atoms are excluded.
    atoms = atoms.loc[((xyz > lower) & (xyz < upper)).all(axis=1)].copy()
    atoms = atoms.loc[atoms["atom_name"].map(accepted_atom_name)].copy()
    if atoms.empty:
        raise ValueError("no accepted protein atoms strictly inside docking box")

    xyz = atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(np.float64)
    centroid = xyz.mean(axis=0)
    # Stable sorting makes equal-distance ties deterministic in original PDB order.
    order = np.argsort(np.linalg.norm(xyz - centroid, axis=1), kind="stable")[:max_atoms]
    selected = atoms.iloc[order].copy().reset_index(drop=True)
    selected["atom_number"] = np.arange(1, len(selected) + 1)
    selected["line_idx"] = np.arange(len(selected))
    metadata = {
        "box_atoms": int(len(atoms)),
        "selected_atoms": int(len(selected)),
        "max_atoms": int(max_atoms),
        "selection": "unimol_strict_box_centroid_topk_v1",
    }
    return selected, metadata


def clean_full_protein(protein_path: str | Path):
    """Apply UniMol atom cleaning while retaining the complete ATOM receptor.

    Unlike :func:`generate_unimol_pocket`, this performs no box cropping,
    centroid sorting, or Top-K truncation. Original PDB atom order is retained.
    """
    atoms = PandasPdb().read_pdb(str(protein_path)).df["ATOM"].copy()
    if atoms.empty:
        raise ValueError("protein contains no ATOM records")
    input_atoms = len(atoms)
    atoms = atoms[
        atoms["residue_name"].astype(str).str.strip().str.upper().isin(STANDARD_AMINO_ACIDS)
    ].copy()
    atoms = atoms[atoms["element_symbol"].astype(str).str.upper() != "H"].copy()
    atoms = atoms.drop_duplicates(ATOM_IDENTITY, keep="first")
    atoms = atoms.loc[atoms["atom_name"].map(accepted_atom_name)].copy()
    if atoms.empty:
        raise ValueError("protein contains no accepted atoms after UniMol cleaning")
    atoms = atoms.reset_index(drop=True)
    atoms["atom_number"] = np.arange(1, len(atoms) + 1)
    atoms["line_idx"] = np.arange(len(atoms))
    return atoms, {
        "input_atom_records": int(input_atoms),
        "selected_atoms": int(len(atoms)),
        "selection": "unimol_atom_cleaning_full_receptor_v1",
    }


def write_unimol_pocket(frame, target: str | Path) -> None:
    """Atomically write a canonical pocket and verify identity/order round-trip."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".ifmdock.tmp.pdb")
    writer = PandasPdb()
    writer.df["ATOM"] = frame
    writer.to_pdb(path=str(temporary), records=["ATOM"], gz=False, append_newline=True)
    reread = PandasPdb().read_pdb(str(temporary)).df["ATOM"]
    if atom_signature(reread) != atom_signature(frame):
        temporary.unlink(missing_ok=True)
        raise ValueError("PDB serialization changed canonical atom order or coordinates")
    os.replace(temporary, target)
