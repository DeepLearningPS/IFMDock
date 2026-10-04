#!/usr/bin/env python3
"""Create deterministic train/validation splits and rigid-pocket PyG caches."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ifmdock.data.feature.featurizer import FeaturizerConfig
from ifmdock.data.modules.training.pipeline import TrainingDataPipeline, TrainingPipelineConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument(
        "--all-train", action="store_true",
        help="Put every eligible complex in train.txt. val.txt mirrors train.txt only "
             "for DataModule construction and must be paired with trainer.limit_val_batches=0.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per-complex-timeout", type=float, default=120.0,
                   help="Seconds before a single parse/featurization task is skipped (0 disables).")
    p.add_argument("--matching", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--matching-popsize", type=int, default=20)
    p.add_argument("--matching-maxiter", type=int, default=20)
    p.add_argument("--require-distance-cache", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if not args.all_train and not 0 < args.val_fraction < 1:
        p.error("--val-fraction must be between zero and one")
    names = []
    for directory in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        name = directory.name
        required = [directory / f"{name}_ligand.sdf", directory / f"{name}_protein_256.pdb"]
        if args.require_distance_cache:
            required.append(directory / f"interaction_{name}_v2.pkl")
        if all(path.is_file() for path in required):
            names.append(name)
    random.Random(args.seed).shuffle(names)
    if args.limit:
        names = names[:args.limit]
    if len(names) < 2:
        raise RuntimeError("fewer than two eligible complexes")
    if args.all_train:
        train = sorted(names)
        # DockingDataModule currently constructs a validation Dataset eagerly.
        # Mirror the list for construction only; the fine-tune config disables
        # every validation batch, so no sample is held out or evaluated.
        val = list(train)
    else:
        n_val = max(1, round(len(names) * args.val_fraction))
        n_val = min(n_val, len(names) - 1)
        val, train = sorted(names[:n_val]), sorted(names[n_val:])
    args.split_dir.mkdir(parents=True, exist_ok=True)
    all_values = train if args.all_train else train + val
    for label, values in (("train", train), ("val", val), ("all", all_values)):
        (args.split_dir / f"{label}.txt").write_text("".join(f"{name}\n" for name in values))
    cfg = TrainingPipelineConfig(
        dataset="rigid_pocket", complex_file=str(args.split_dir / "all.txt"),
        data_dir=str(args.dataset), cache_path=str(args.cache),
        apo_protein_file="protein_256", holo_protein_file="protein_256",
        num_workers=args.workers, task_timeout_seconds=args.per_complex_timeout,
    )
    features = FeaturizerConfig(
        matching=args.matching,
        popsize=args.matching_popsize if args.matching else None,
        maxiter=args.matching_maxiter if args.matching else None,
        keep_original=True,
        remove_hs=True, num_conformers=1, flexible_backbone=False,
        flexible_sidechains=False, rigid_pocket=True,
    )
    TrainingDataPipeline(cfg, features).process_all_complexes()
    def actually_cached(values):
        return [name for name in values
                if (args.cache / f"heterograph-{name}-0.pt").is_file()
                and (not args.require_distance_cache or
                     (args.dataset / name / f"interaction_{name}_v2.pkl").is_file())]
    train_kept, val_kept = actually_cached(train), actually_cached(val)
    if not train_kept or not val_kept:
        raise RuntimeError(
            f"graph preprocessing left an empty split: train={len(train_kept)} val={len(val_kept)}"
        )
    all_kept = train_kept if args.all_train else train_kept + val_kept
    for label, values in (("train", train_kept), ("val", val_kept),
                          ("all", all_kept)):
        (args.split_dir / f"{label}.txt").write_text("".join(f"{name}\n" for name in values))
    excluded = sorted(set(names) - set(train_kept) - set(val_kept))
    (args.split_dir / "excluded_after_graph_preprocessing.txt").write_text(
        "".join(f"{name}\n" for name in excluded)
    )
    summary = {"eligible_before_graphs": len(names), "train": len(train_kept),
               "val": len(val_kept), "excluded_after_graphs": len(excluded),
               "validation_disabled": bool(args.all_train),
               "seed": args.seed, "cache": str(args.cache)}
    (args.split_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
