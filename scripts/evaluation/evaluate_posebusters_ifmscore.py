#!/usr/bin/env python3
"""IFMScore ranking and PoseBusters validity for IFMDock PoseBusters samples.

Run ``score`` once per GPU shard under the IFMScore environment, then run
``finalize`` once after all shards finish.  Per-complex files make both stages
restartable without repeating completed work.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from ifmdock.utils.protein_paths import resolve_posebusters_protein


def _raise_posebusters_timeout(_signum, _frame):
    raise TimeoutError("PoseBusters validation timed out")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--epoch-dir", type=Path, required=True)
    score.add_argument("--data-dir", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--shard-index", type=int, required=True)
    score.add_argument("--num-shards", type=int, required=True)
    score.add_argument("--batch-size", type=int, default=128)
    score.add_argument(
        "--protein-mode",
        choices=("reference10", "origin_standard_aa", "ifmdock_clean", "unimol256"),
        default="reference10",
        help=("reference10: derive IFMScore's 10 A pocket from the cleaned working "
              "protein; origin_standard_aa: use the origin receptor with only "
              "nonstandard amino acids removed; ifmdock_clean: use the standalone "
              "clean receptor; unimol256: use *_protein_256.pdb directly."),
    )
    score.add_argument("--force", action="store_true")
    score.add_argument(
        "--only-failed", action="store_true",
        help=("Score only complexes whose existing per-complex IFMScore record "
              "is missing or has a non-ok status. Useful for protein-input fallback."),
    )

    final = subparsers.add_parser("finalize")
    final.add_argument("--epoch-dir", type=Path, required=True)
    final.add_argument("--data-dir", type=Path, required=True)
    final.add_argument("--workers", type=int, default=16)
    final.add_argument("--force-posebusters", action="store_true")
    final.add_argument(
        "--posebusters-timeout", type=int, default=120, metavar="SECONDS",
        help=("Maximum time per complex for the selected-pose PoseBusters check; "
              "0 disables the timeout. Timed-out complexes are counted as failures."),
    )
    final.add_argument(
        "--posebusters-protein-mode",
        choices=("origin_priority", "cleaned"), default="origin_priority",
        help=("Conditioning receptor for final Top-1/Best-1 PoseBusters: "
              "origin_priority uses EC-Dock priority; cleaned requires <id>_protein.pdb."),
    )
    final.add_argument(
        "--use-cached-posebusters", action="store_true",
        help=("Do not rerun Top-1 PoseBusters; derive physical validity from the "
              "candidate-wise posebusters_all masks already present in epoch-dir."),
    )
    final.add_argument(
        "--check-best-and-ifmscore-top1", action="store_true",
        help=("Run PoseBusters only for the RMSD Best-1 and IFMScore Top-1 "
              "candidates (one check if both select the same pose)."),
    )
    final.add_argument("--skip-posebusters", action="store_true",
                       help="Do not run physical checks; physical metrics are null.")
    return parser.parse_args()


def score_complex(
    model, device, ligand_sdf, protein_pdb, reference_sdf, batch_size,
    generate_reference_pocket=True,
):
    from score import score_with_model

    ids, predictions = score_with_model(
        model, protein_pdb, ligand_sdf, cutoff=10.0,
        reference=reference_sdf if generate_reference_pocket else None,
        batch_size=batch_size, device=device,
    )
    # VSDataset IDs end in the original SDF record number.  Preserve that
    # mapping if an individual molecule failed graph construction.
    sample_indices = [int(str(item).rsplit("-", 1)[-1]) for item in ids]
    scores = np.asarray(predictions, dtype=float).reshape(-1)
    if len(sample_indices) != len(scores):
        raise ValueError(f"id/score mismatch: {len(sample_indices)} != {len(scores)}")
    return sample_indices, scores


def run_score(args):
    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "IFMScore"))
    from score import load_model
    inputs = pd.read_csv(args.epoch_dir / "inputs.csv")
    rows = inputs.iloc[args.shard_index :: args.num_shards]
    output_root = args.epoch_dir / "ifmscore"
    output_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device=device)
    succeeded = failed = 0
    for sequence, row in enumerate(rows.itertuples(index=False), 1):
        name = str(row.pdbid)
        target = output_root / f"{name}.json"
        if args.only_failed and target.is_file():
            try:
                if json.loads(target.read_text()).get("status") == "ok":
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        if target.is_file() and not args.force:
            try:
                if json.loads(target.read_text()).get("status") == "ok":
                    succeeded += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        ligand = args.epoch_dir / "predictions" / name / f"gen_{name}_ligand.sdf"
        reference = args.data_dir / name / f"{name}_ligand.sdf"
        if args.protein_mode == "unimol256":
            protein = args.data_dir / name / f"{name}_protein_256.pdb"
        elif args.protein_mode == "origin_standard_aa":
            protein = args.data_dir / name / f"{name}_protein_origin_standard_aa.pdb"
        elif args.protein_mode == "ifmdock_clean":
            protein = args.data_dir / name / f"{name}_protein_ifmdock_clean.pdb"
        else:
            protein = args.data_dir / name / f"{name}_protein.pdb"
        record = {
            "complex_id": name, "status": "failed",
            "protein_mode": args.protein_mode,
        }
        try:
            indices, scores = score_complex(
                model, device, ligand, protein, reference, args.batch_size,
                generate_reference_pocket=args.protein_mode != "unimol256",
            )
            if not len(scores):
                raise ValueError("no IFMScore predictions")
            best = int(np.argmax(scores))
            record.update(
                status="ok",
                scored_poses=len(scores),
                sample_indices=indices,
                scores=scores.tolist(),
                top1_sample_index=int(indices[best]),
                top1_score=float(scores[best]),
            )
            succeeded += 1
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            failed += 1
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, target)
        print(
            f"shard={args.shard_index} {sequence}/{len(rows)} {name} "
            f"{record['status']}", flush=True
        )
    print(json.dumps({"shard": args.shard_index, "ok": succeeded, "failed": failed}))


def posebusters_selected(task):
    (name, ifmscore_index, best_index, epoch_dir, data_dir, force, timeout,
     protein_mode) = task
    epoch_dir, data_dir = Path(epoch_dir), Path(data_dir)
    output_dir = epoch_dir / "posebusters_top1" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}_selected_posebusters.csv"
    json_path = output_dir / f"{name}_selected_validity.json"
    if json_path.is_file() and not force:
        try:
            return json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    requested = {
        "ifmscore_top1": int(ifmscore_index),
        "best1": int(best_index),
    }
    result = {"complex_id": name, "status": "failed", "indices": requested}
    if timeout > 0:
        signal.signal(signal.SIGALRM, _raise_posebusters_timeout)
        signal.alarm(timeout)
    try:
        from posebusters import PoseBusters

        sdf = epoch_dir / "predictions" / name / f"gen_{name}_ligand.sdf"
        poses = [mol for mol in Chem.SDMolSupplier(str(sdf), removeHs=False) if mol]
        unique_indices = list(dict.fromkeys(requested.values()))
        if any(index < 0 or index >= len(poses) for index in unique_indices):
            raise IndexError(f"selected indices {unique_indices} outside {len(poses)} poses")
        reference = Chem.SDMolSupplier(
            str(data_dir / name / f"{name}_ligand.sdf"), removeHs=False
        )[0]
        if protein_mode == "cleaned":
            protein_path = data_dir / name / f"{name}_protein.pdb"
            if not protein_path.is_file():
                raise FileNotFoundError(f"missing cleaned conditioning protein: {protein_path}")
        else:
            protein_path = resolve_posebusters_protein(data_dir / name, name)
        result["conditioning_protein"] = str(protein_path)
        protein = Chem.MolFromPDBFile(str(protein_path), removeHs=False, sanitize=False)
        if protein is None:
            raise ValueError(f"conditioning protein could not be read: {protein_path}")
        table = PoseBusters("dock").bust(
            mol_pred=[poses[index] for index in unique_indices],
            mol_true=reference, mol_cond=protein,
        )
        table.to_csv(csv_path)
        if len(table) != len(unique_indices):
            raise ValueError(
                f"PoseBusters returned {len(table)} rows for {len(unique_indices)} poses"
            )
        checks_by_index, valid_by_index = {}, {}
        for row_index, sample_index in enumerate(unique_indices):
            row = table.iloc[row_index]
            checks = {
                str(column): bool(False if pd.isna(row[column]) else row[column])
                for column in table.columns
            }
            checks_by_index[str(sample_index)] = checks
            valid_by_index[str(sample_index)] = bool(all(checks.values()))
        result.update(
            status="ok", valid_by_index=valid_by_index,
            checks_by_index=checks_by_index,
            ifmscore_top1_physically_valid=valid_by_index[str(ifmscore_index)],
            best1_physically_valid=valid_by_index[str(best_index)],
            checked_pose_count=len(unique_indices), csv_file=str(csv_path),
        )
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            result["status"] = "timed_out"
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if timeout > 0:
            signal.alarm(0)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, json_path)
    return result


def run_finalize(args):
    rmsd_rows = json.loads((args.epoch_dir / "per_complex_rmsd.json").read_text())
    rmsd_by_name = {row["complex_id"]: row for row in rmsd_rows}
    score_records = {}
    for path in sorted((args.epoch_dir / "ifmscore").glob("*.json")):
        try:
            row = json.loads(path.read_text())
            score_records[row["complex_id"]] = row
        except (OSError, json.JSONDecodeError, KeyError):
            continue

    tasks = []
    for name, score in score_records.items():
        if (score.get("status") == "ok" and not args.skip_posebusters
                and not args.use_cached_posebusters):
            rmsd = rmsd_by_name.get(name, {})
            values = rmsd.get("rmsds", []) if rmsd.get("status") == "ok" else []
            if not values:
                continue
            best_index = int(np.argmin(np.asarray(values, dtype=float)))
            tasks.append((
                name, int(score["top1_sample_index"]), best_index,
                str(args.epoch_dir), str(args.data_dir), args.force_posebusters,
                args.posebusters_timeout, args.posebusters_protein_mode,
            ))
    validity = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(posebusters_selected, task): task[0] for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                validity[row["complex_id"]] = row
                print(f"PoseBusters {len(validity)}/{len(tasks)} {row['complex_id']} {row['status']}", flush=True)

    inputs = pd.read_csv(args.epoch_dir / "inputs.csv")
    details = []
    for name in inputs["pdbid"].astype(str):
        rmsd = rmsd_by_name.get(name, {})
        score = score_records.get(name, {})
        valid = validity.get(name, {})
        all_mask = None
        all_mask_path = args.epoch_dir / "posebusters_all" / f"{name}.json"
        if not args.skip_posebusters and all_mask_path.is_file():
            try:
                all_mask = json.loads(all_mask_path.read_text()).get("valid_mask")
            except (OSError, json.JSONDecodeError):
                all_mask = None
        row = {
            "complex_id": name,
            "rmsd_status": rmsd.get("status", "missing"),
            "ifmscore_status": score.get("status", "missing"),
            "posebusters_status": (
                "cached" if args.use_cached_posebusters and all_mask is not None
                else valid.get("status", "missing")
            ),
        }
        if rmsd.get("status") == "ok":
            values = rmsd["rmsds"]
            row.update(
                best_rmsd=float(rmsd["best_rmsd"]),
                mean_rmsd=float(rmsd["mean_rmsd"]),
                best1_pass=bool(rmsd["best_rmsd"] <= 2.0),
                average_pass=bool(rmsd["mean_rmsd"] <= 2.0),
            )
            if all_mask is not None:
                physical_indices = [i for i, passed in enumerate(all_mask)
                                    if bool(passed) and i < len(values)]
                row["best1_physical_pass"] = bool(
                    physical_indices and min(float(values[i]) for i in physical_indices) <= 2.0
                )
            if args.check_best_and_ifmscore_top1 and not args.skip_posebusters:
                best_physical = valid.get("best1_physically_valid")
                row["best1_physically_valid"] = best_physical
                row["best1_physical_pass"] = bool(
                    best_physical is True and float(rmsd["best_rmsd"]) <= 2.0
                )
            if score.get("status") == "ok":
                index = int(score["top1_sample_index"])
                if index < len(values):
                    top1_rmsd = float(values[index])
                    if args.skip_posebusters:
                        physical = None
                    elif args.use_cached_posebusters:
                        physical = bool(
                            all_mask is not None and index < len(all_mask) and all_mask[index]
                        )
                    else:
                        physical = valid.get("ifmscore_top1_physically_valid")
                    row.update(
                        ifmscore_top1_index=index,
                        ifmscore_top1_score=float(score["top1_score"]),
                        ifmscore_top1_rmsd=top1_rmsd,
                        ifmscore_top1_pass=bool(top1_rmsd <= 2.0),
                        ifmscore_top1_physically_valid=physical,
                        ifmscore_top1_physical_pass=(None if args.skip_posebusters else bool(
                            physical is True and top1_rmsd <= 2.0
                        )),
                    )
        details.append(row)

    evaluable = [row for row in details if row.get("rmsd_status") == "ok"]
    ranked = [row for row in evaluable if "ifmscore_top1_pass" in row]
    physical_evaluable = [
        row for row in ranked if row.get("ifmscore_top1_physically_valid") is not None
    ]

    def rate(rows, key):
        return float(np.mean([bool(row.get(key, False)) for row in rows])) if rows else None

    summary = {
        "requested_complexes": len(details),
        "rmsd_evaluable_complexes": len(evaluable),
        "ifmscore_ranked_complexes": len(ranked),
        "posebusters_evaluable_complexes": len(physical_evaluable),
        "rmsd_threshold_angstrom": 2.0,
        "best1_pass_count": sum(row.get("best1_pass", False) for row in evaluable),
        "best1_rate": rate(evaluable, "best1_pass"),
        "ifmscore_top1_pass_count": sum(row.get("ifmscore_top1_pass", False) for row in ranked),
        "ifmscore_top1_rate": rate(ranked, "ifmscore_top1_pass"),
        "average_pass_count": sum(row.get("average_pass", False) for row in evaluable),
        "average_rate": rate(evaluable, "average_pass"),
        "best1_physical_pass_count": None if args.skip_posebusters else sum(
            row.get("best1_physical_pass", False) for row in evaluable),
        "best1_physical_rate": None if args.skip_posebusters else rate(
            details, "best1_physical_pass"),
        "ifmscore_top1_physical_pass_count": None if args.skip_posebusters else sum(
            row.get("ifmscore_top1_physical_pass", False) for row in ranked
        ),
        # Match ECDock's strict convention: a PoseBusters execution/parsing
        # failure is not removed from the denominator; it is an invalid pose.
        "ifmscore_top1_physical_rate": None if args.skip_posebusters else rate(ranked, "ifmscore_top1_physical_pass"),
        "ifmscore_top1_physically_valid_rate": None if args.skip_posebusters else rate(
            ranked, "ifmscore_top1_physically_valid"
        ),
        "strict_rates_over_all_requested": {
            "best1": rate(details, "best1_pass"),
            "ifmscore_top1": rate(details, "ifmscore_top1_pass"),
            "average": rate(details, "average_pass"),
            "best1_physical": None if args.skip_posebusters else rate(details, "best1_physical_pass"),
            "ifmscore_top1_physical": None if args.skip_posebusters else rate(details, "ifmscore_top1_physical_pass"),
        },
        "mean_best_rmsd": float(np.mean([row["best_rmsd"] for row in evaluable])) if evaluable else None,
        "mean_of_40_pose_mean_rmsd": float(np.mean([row["mean_rmsd"] for row in evaluable])) if evaluable else None,
        "mean_ifmscore_top1_rmsd": float(np.mean([row["ifmscore_top1_rmsd"] for row in ranked])) if ranked else None,
        "ifmscore_model": "ifmscore.pth",
        "physical_validity": ("not evaluated (--skip-posebusters)" if args.skip_posebusters else
                              "all PoseBusters dock checks pass"),
        "posebusters_protein_mode": args.posebusters_protein_mode,
    }
    summary["five_metrics_over_all_requested"] = {
        "ifmscore_top1": rate(details, "ifmscore_top1_pass"),
        "ifmscore_top1_physical": None if args.skip_posebusters else rate(
            details, "ifmscore_top1_physical_pass"),
        "best_top1": rate(details, "best1_pass"),
        "best_top1_physical": None if args.skip_posebusters else rate(
            details, "best1_physical_pass"),
        "average_rmsd": rate(details, "average_pass"),
        "denominator": len(details),
    }
    (args.epoch_dir / "full_evaluation_per_complex.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )
    pd.DataFrame(details).to_csv(args.epoch_dir / "full_evaluation_per_complex.csv", index=False)
    (args.epoch_dir / "full_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.command == "score":
        run_score(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
