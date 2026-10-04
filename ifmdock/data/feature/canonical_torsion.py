import numpy as np
import torch


def chemical_torsion_periods(graph, mol=None):
    n = int(graph["ligand"].edge_mask.sum())
    periods = torch.full((n,), 2 * np.pi, dtype=torch.float32)
    mol = mol if mol is not None else getattr(graph, "mol", None)
    if mol is None or mol.GetNumAtoms() != int(graph["ligand"].num_nodes):
        return periods
    from rdkit import Chem
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    edges = graph["ligand", "lig_bond", "ligand"].edge_index
    for i, (fixed, rotating) in enumerate(edges[:, graph["ligand"].edge_mask].T.tolist()):
        values = [ranks[a.GetIdx()] for a in mol.GetAtomWithIdx(rotating).GetNeighbors()
                  if a.GetIdx() != fixed]
        multiplicity = max((values.count(v) for v in set(values)), default=1)
        if multiplicity in (2, 3):
            periods[i] = 2 * np.pi / multiplicity
    return periods


def attach_canonical_torsions(graph, mol=None):
    mol = mol if mol is not None else getattr(graph, "mol", None)
    if mol is None or mol.GetNumAtoms() != int(graph["ligand"].num_nodes):
        raise ValueError("canonical torsions require an atom-aligned RDKit molecule")
    from rdkit import Chem
    ranks = list(Chem.CanonicalRankAtoms(mol, includeChirality=True, breakTies=True))
    edges = graph["ligand", "lig_bond", "ligand"].edge_index
    tuples = []
    for j, k in edges[:, graph["ligand"].edge_mask].T.tolist():
        left = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        right = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
        if not left or not right:
            raise ValueError(f"rotatable bond {j}-{k} lacks canonical reference atoms")
        tuples.append((min(left, key=lambda a: (ranks[a], a)), j, k,
                       min(right, key=lambda a: (ranks[a], a))))
    graph["ligand"].canonical_torsion_index = (
        torch.tensor(tuples, dtype=torch.long).T.contiguous()
        if tuples else torch.empty((4, 0), dtype=torch.long)
    )
    graph["ligand"].canonical_tor_period = chemical_torsion_periods(graph, mol=mol)
    return graph
