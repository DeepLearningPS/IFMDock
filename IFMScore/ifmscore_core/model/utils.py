"""Inference helpers for IFMScore."""

import torch
from torch.distributions import Normal
from torch_scatter import scatter_add


def run_an_eval_epoch(model, data_loader, pred=True, dist_threhold=5.0, device="cpu"):
    """Score each protein-ligand graph in a PyG loader."""
    if not pred:
        raise ValueError("IFMScore exposes prediction only")
    model.eval()
    predictions = []
    with torch.no_grad():
        for pdbids, protein_graph, ligand_graph, _labels in data_loader:
            protein_graph = protein_graph.to(device)
            ligand_graph = ligand_graph.to(device)
            pi, sigma, mu, distances, _atom_types, _bond_types, batch = model(
                ligand_graph, protein_graph
            )
            normal = Normal(mu, sigma)
            log_probability = normal.log_prob(distances.expand_as(normal.loc))
            probability = (log_probability + torch.log(pi)).exp().sum(dim=1)
            if dist_threhold is not None:
                probability = probability.masked_fill(
                    distances.squeeze(-1) > dist_threhold, 0.0
                )
            batch = batch.to(device)
            graph_count = int(torch.unique(batch).numel())
            predictions.append(scatter_add(probability, batch, dim=0, dim_size=graph_count))
    return torch.cat(predictions).cpu().numpy()
