"""Standalone IFMDock benchmark metrics for output/<complex_id>/.

Adapted conceptually from EcDock_Evaluate/evaluate.py, docking_evaluate.py,
and JS_KL/. This module does not modify poses or filter by PoseBusters.

RMSD is a fixed-frame, heavy-atom RMSD over symmetry-compatible mappings.
The receptor and crystal ligand must use the same coordinate frame as poses.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde


RMSD_CUTOFF = 2.0


def read_sdf(path: Path) -> list[Chem.Mol]:
    return [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol]


def docking_rmsd(pose: Chem.Mol, crystal: Chem.Mol) -> float:
    """Minimum unaligned RMSD over matching heavy-atom graph automorphisms.

    `GetBestRMS` aligns coordinates and would conceal docking translation or
    rotation errors, so it must not be used for the main docking metric.
    """
    pose, crystal = Chem.RemoveHs(pose), Chem.RemoveHs(crystal)
    if pose.GetNumAtoms() != crystal.GetNumAtoms():
        raise ValueError("pose and crystal have different heavy-atom counts")
    matches = crystal.GetSubstructMatches(pose, uniquify=False, maxMatches=100000)
    if not matches:
        raise ValueError("pose and crystal heavy-atom graphs do not match")
    xyz = np.asarray(pose.GetConformer().GetPositions())
    reference_xyz = np.asarray(crystal.GetConformer().GetPositions())
    return float(min(np.sqrt(np.sum((xyz - reference_xyz[list(mapping)]) ** 2) / len(xyz))
                     for mapping in matches))


def pb_valid(pose: Chem.Mol, crystal: Chem.Mol, protein: Chem.Mol) -> bool:
    """All PoseBusters dock checks must pass; RMSD success is reported separately."""
    from posebusters import PoseBusters

    report = PoseBusters("dock").bust(mol_pred=pose, mol_true=crystal, mol_cond=protein)
    if len(report) != 1:
        raise ValueError("PoseBusters did not return exactly one row")
    checks = report.select_dtypes(include=["bool"])
    if checks.shape[1] == 0:
        raise ValueError("PoseBusters returned no boolean checks")
    return bool(checks.iloc[0].all())


def evaluate_complex(directory: Path, run_pb: bool = False) -> dict:
    """Top-1 is the first IFMScore-ranked SDF record; Best-1 has lowest RMSD."""
    name = directory.name
    poses = read_sdf(directory / f"gen_{name}_ligand.sdf")
    crystal_mols = read_sdf(directory / f"{name}_ligand.sdf")
    if not poses or not crystal_mols:
        raise ValueError("missing valid generated pose or crystal ligand")
    crystal = crystal_mols[0]
    rmsds = [docking_rmsd(pose, crystal) for pose in poses]
    best_index = int(np.argmin(rmsds))
    row = {
        "complex_id": name,
        "pose_count": len(poses),
        "pose_rmsds": rmsds,
        "top1_rmsd": rmsds[0],
        "best1_rmsd": rmsds[best_index],
        "best1_pose_index": best_index,
        "mean_rmsd": float(np.mean(rmsds)),
        "top1_success": rmsds[0] <= RMSD_CUTOFF,
        "best1_success": rmsds[best_index] <= RMSD_CUTOFF,
        "mrsr_success": float(np.mean(rmsds)) <= RMSD_CUTOFF,
        "top1_pb_valid": None,
        "best1_pb_valid": None,
    }
    if run_pb:
        protein = Chem.MolFromPDBFile(
            str(directory / f"{name}_protein.pdb"), removeHs=False, sanitize=False
        )
        if protein is None:
            raise ValueError("could not read conditioning protein")
        row["top1_pb_valid"] = pb_valid(poses[0], crystal, protein)
        row["best1_pb_valid"] = (
            row["top1_pb_valid"] if best_index == 0
            else pb_valid(poses[best_index], crystal, protein)
        )
    return row


def summarize(rows: list[dict]) -> dict:
    """Each complex has equal weight; failed complexes should be reported separately."""
    if not rows:
        raise ValueError("no evaluable complexes")
    avg = lambda key: float(np.mean([row[key] for row in rows]))
    result = {
        "complexes_evaluated": len(rows),
        "top1_rmsd_mean": avg("top1_rmsd"),
        "best1_rmsd_mean": avg("best1_rmsd"),
        "mean_pose_rmsd": avg("mean_rmsd"),
        "top1_success_rate": avg("top1_success"),
        "best1_success_rate": avg("best1_success"),
        "mrsr": avg("mrsr_success"),
    }
    if all(row["top1_pb_valid"] is not None for row in rows):
        result.update({
            "top1_pb_valid_rate": avg("top1_pb_valid"),
            "best1_pb_valid_rate": avg("best1_pb_valid"),
            "top1_success_and_pb_valid_rate": float(np.mean([
                row["top1_success"] and row["top1_pb_valid"] for row in rows
            ])),
            "best1_success_and_pb_valid_rate": float(np.mean([
                row["best1_success"] and row["best1_pb_valid"] for row in rows
            ])),
        })
    return result


def geometry_values(mol: Chem.Mol) -> dict[str, dict[str, list[float]]]:
    """Collect bond lengths, angles and dihedrals by local chemical type.

    These generic type keys are consistent within this evaluator. They are not
    identical to the hand-picked SMARTS categories in the original JS_KL files.
    """
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    values = {kind: defaultdict(list) for kind in ("bond_length", "bond_angle", "dihedral")}

    def atom_label(atom: Chem.Atom) -> str:
        return atom.GetSymbol()

    def bond_label(bond: Chem.Bond) -> str:
        return str(bond.GetBondType())

    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ends = sorted((atom_label(mol.GetAtomWithIdx(a)), atom_label(mol.GetAtomWithIdx(b))))
        key = f"{ends[0]}:{bond_label(bond)}:{ends[1]}"
        values["bond_length"][key].append(rdMolTransforms.GetBondLength(conf, a, b))

    for center in mol.GetAtoms():
        j = center.GetIdx()
        neighbors = [atom.GetIdx() for atom in center.GetNeighbors()]
        for left in range(len(neighbors)):
            for right in range(left + 1, len(neighbors)):
                i, k = neighbors[left], neighbors[right]
                sides = sorted((atom_label(mol.GetAtomWithIdx(i)), atom_label(mol.GetAtomWithIdx(k))))
                key = f"{sides[0]}-{atom_label(center)}-{sides[1]}"
                values["bond_angle"][key].append(rdMolTransforms.GetAngleDeg(conf, i, j, k))

    for central in mol.GetBonds():
        j, k = central.GetBeginAtomIdx(), central.GetEndAtomIdx()
        left = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        right = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
        for i in left:
            for l in right:
                atoms = [atom_label(mol.GetAtomWithIdx(n)) for n in (i, j, k, l)]
                forward = "-".join(atoms)
                key = min(forward, "-".join(reversed(atoms)))
                values["dihedral"][key].append(rdMolTransforms.GetDihedralDeg(conf, i, j, k, l))
    return values


def js_distance(reference: list[float], generated: list[float], *, points: int = 100,
                epsilon: float = 1e-6) -> float:
    """JS_KL convention: KDE on a shared grid, then SciPy's JS *distance*.

    SciPy jensenshannon returns sqrt(JS divergence). Square this value if a
    mathematically defined JS divergence is needed instead of the source metric.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(generated, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("KDE needs at least two varying values per distribution")
    grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), points)
    p = gaussian_kde(a)(grid) + epsilon
    q = gaussian_kde(b)(grid) + epsilon
    return float(jensenshannon(p / p.sum(), q / q.sum()))


def geometry_jsd(directories: list[Path]) -> dict:
    """Pool Top-1 generated and crystal geometries; average valid type scores."""
    pooled = {kind: (defaultdict(list), defaultdict(list))
              for kind in ("bond_length", "bond_angle", "dihedral")}
    for directory in directories:
        name = directory.name
        generated = read_sdf(directory / f"gen_{name}_ligand.sdf")[0]
        crystal = read_sdf(directory / f"{name}_ligand.sdf")[0]
        for kind, by_type in geometry_values(generated).items():
            for key, items in by_type.items():
                pooled[kind][0][key].extend(items)
        for kind, by_type in geometry_values(crystal).items():
            for key, items in by_type.items():
                pooled[kind][1][key].extend(items)
    result = {}
    for kind, (generated, crystal) in pooled.items():
        scores = {}
        for key in sorted(generated.keys() & crystal.keys()):
            try:
                scores[key] = js_distance(crystal[key], generated[key])
            except (ValueError, np.linalg.LinAlgError):
                continue
        result[kind] = {"by_type": scores,
                        "mean": float(np.mean(list(scores.values()))) if scores else None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="IFMDock output/ directory")
    parser.add_argument("--posebusters", action="store_true", help="run PB checks on Top-1 and Best-1")
    parser.add_argument("--geometry-jsd", action="store_true", help="compare Top-1 and crystal geometry")
    args = parser.parse_args()
    rows, failures, valid_dirs = [], {}, []
    for directory in sorted(p for p in args.output.iterdir() if p.is_dir()):
        if not (directory / f"gen_{directory.name}_ligand.sdf").is_file():
            continue
        try:
            rows.append(evaluate_complex(directory, args.posebusters))
            valid_dirs.append(directory)
        except Exception as error:
            failures[directory.name] = str(error)
    report = {"summary": summarize(rows), "per_complex": rows, "failures": failures}
    if args.geometry_jsd:
        report["geometry_jsd"] = geometry_jsd(valid_dirs)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
