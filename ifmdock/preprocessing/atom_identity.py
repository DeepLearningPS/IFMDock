"""Coordinate-free atom identities and exact index permutations."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from biopandas.pdb import PandasPdb

from ifmdock.data.parse.protein import parse_pdb_from_path
from ifmdock.preprocessing.unimol_pocket import accepted_atom_name


def protein_atom_id(atom) -> str:
    residue = atom.residue
    chain = str(getattr(residue, "chain", "") or "").strip()
    number = str(getattr(residue, "number", "")).strip()
    insertion = str(getattr(residue, "insertion_code", "") or "").strip()
    altloc = str(getattr(atom, "altloc", "") or "").strip()
    return f"{chain}|{number}|{insertion}|{str(atom.name).strip()}|{altloc}"


def pyg_protein_atom_ids(pdb_path: str | Path) -> list[str]:
    """Return identities in the exact ParmEd order consumed by IFMDock/PyG."""
    struct = parse_pdb_from_path(str(pdb_path), remove_hs=True, reorder=True)
    ids = [protein_atom_id(atom) for atom in struct.atoms]
    assert_unique(ids, "protein")
    return ids


def distance_protein_atom_ids(pdb_path: str | Path) -> list[str]:
    """Return the row identities retained by the UniMol/EC-Dock processor."""
    frame = PandasPdb().read_pdb(str(pdb_path)).df["ATOM"].copy()
    frame = frame[frame["element_symbol"].astype(str).str.upper() != "H"].copy()
    # This exactly mirrors the validated processor: first altloc wins for a
    # physical atom identity, then unsupported atom names are removed.
    physical = ["chain_id", "residue_number", "insertion", "atom_name"]
    frame = frame.drop_duplicates(physical, keep="first")
    frame = frame.loc[frame["atom_name"].map(accepted_atom_name)]
    ids = [
        f"{str(row.chain_id).strip()}|{str(row.residue_number).strip()}|"
        f"{str(row.insertion).strip()}|{str(row.atom_name).strip()}|"
        f"{str(getattr(row, 'alt_loc', '')).strip()}"
        for row in frame.itertuples(index=False)
    ]
    assert_unique(ids, "distance protein")
    return ids


def ligand_atom_ids(sdf_path: str | Path) -> list[str]:
    """Stable graph identities in source-SDF heavy-atom order.

    Unique atom-map numbers are authoritative. Otherwise a Weisfeiler-Lehman
    style local graph signature plus a deterministic occurrence counter is
    used. The complete identity list is persisted on both sides of the
    permutation and therefore never depends on conformer coordinates.
    """
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=True)
    if mol is None:
        raise ValueError(f"RDKit failed to read {sdf_path}")
    maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if maps and all(value > 0 for value in maps) and len(set(maps)) == len(maps):
        return [f"map:{value}" for value in maps]

    labels = [
        f"{a.GetAtomicNum()}:{a.GetIsotope()}:{a.GetFormalCharge()}:"
        f"{int(a.GetIsAromatic())}:{a.GetTotalDegree()}"
        for a in mol.GetAtoms()
    ]
    for _ in range(4):
        refined = []
        for atom in mol.GetAtoms():
            neighbours = sorted(
                f"{bond.GetBondTypeAsDouble():g}:{int(bond.GetIsAromatic())}:"
                f"{labels[bond.GetOtherAtomIdx(atom.GetIdx())]}"
                for bond in atom.GetBonds()
            )
            refined.append(labels[atom.GetIdx()] + "[" + ";".join(neighbours) + "]")
        labels = refined
    occurrence = defaultdict(int)
    result = []
    for label in labels:
        result.append(f"graph:{label}#{occurrence[label]}")
        occurrence[label] += 1
    assert_unique(result, "ligand")
    return result


def assert_unique(ids: list[str], label: str) -> None:
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"non-unique {label} atom identities: {duplicates[:5]}")


def identity_permutation(source_ids: list[str], target_ids: list[str]) -> list[int]:
    """Map every source row/column to its target PyG index."""
    assert_unique(source_ids, "source")
    assert_unique(target_ids, "target")
    if set(source_ids) != set(target_ids):
        missing = sorted(set(source_ids) - set(target_ids))[:5]
        extra = sorted(set(target_ids) - set(source_ids))[:5]
        raise ValueError(f"atom identity mismatch: source_only={missing}, target_only={extra}")
    target = {identity: index for index, identity in enumerate(target_ids)}
    return [target[identity] for identity in source_ids]
