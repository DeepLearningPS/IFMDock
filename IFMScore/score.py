"""Score ligand poses against a protein pocket with IFMScore."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifmscore_core.data.data import VSDataset
from ifmscore_core.model.model import IFMScore, GatedGCN, GraphTransformer
from ifmscore_core.model.utils import run_an_eval_epoch


def load_model(model_path, *, encoder="gt", device=None):
    """Load the IFMScore model once for repeated scoring calls."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = 128
    if encoder == "gt":
        def build_encoder(edge_dim):
            return GraphTransformer(
                in_channels=41, edge_features=edge_dim, num_hidden_channels=node_dim,
                activ_fn=torch.nn.SiLU(), transformer_residual=True,
                num_attention_heads=4, norm_to_apply="batch", dropout_rate=0.15,
                num_layers=6,
            )
        ligand_encoder, protein_encoder = build_encoder(10), build_encoder(5)
    elif encoder == "gatedgcn":
        def build_encoder(edge_dim):
            return GatedGCN(
                in_channels=41, edge_features=edge_dim, num_hidden_channels=node_dim,
                residual=True, dropout_rate=0.15, equivstable_pe=False, num_layers=6,
            )
        ligand_encoder, protein_encoder = build_encoder(10), build_encoder(5)
    else:
        raise ValueError("encoder must be 'gt' or 'gatedgcn'")

    model = IFMScore(
        ligand_encoder, protein_encoder, in_channels=node_dim, hidden_dim=128,
        n_gaussians=10, dropout_rate=0.15, dist_threhold=5.0,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score_with_model(model, protein, ligand, *, cutoff=10.0, reference=None,
                     batch_size=128, workers=0, device=None):
    """Return (ligand IDs, scores) using an already loaded model."""
    device = device or next(model.parameters()).device
    dataset = VSDataset(
        ligs=str(ligand), prot=str(protein), cutoff=cutoff,
        gen_pocket=reference is not None,
        reflig=str(reference) if reference is not None else None,
        explicit_H=False, use_chirality=True, parallel=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    scores = run_an_eval_epoch(model, loader, pred=True, dist_threhold=5.0, device=device)
    return dataset.ids, scores


def score(protein, ligand, model_path, *, cutoff=10.0, encoder="gt", batch_size=128,
          workers=0, device=None):
    """Return (ligand IDs, scores) for an SDF/MOL2 ligand file."""
    model = load_model(model_path, encoder=encoder, device=device)
    return score_with_model(
        model, protein, ligand, cutoff=cutoff, batch_size=batch_size,
        workers=workers, device=device,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", required=True, help="Protein or pre-cut pocket PDB")
    parser.add_argument("--ligand", required=True, help="Ligand poses in SDF or MOL2 format")
    parser.add_argument(
        "--model", default=str(Path(__file__).parent / "trained_models/ifmscore.pth"),
        help="IFMScore checkpoint",
    )
    parser.add_argument("--encoder", choices=("gt", "gatedgcn"), default="gt")
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--output", default="ifmscore.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    ids, values = score(
        args.protein, args.ligand, args.model, cutoff=args.cutoff,
        encoder=args.encoder, batch_size=args.batch_size, workers=args.workers,
    )
    result = pd.DataFrame({"id": ids, "score": values}).sort_values("score", ascending=False)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
