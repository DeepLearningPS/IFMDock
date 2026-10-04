#!/usr/bin/env python3
"""Sample and score the fixed-pocket PoseBusters benchmark for one epoch.

The script deliberately runs outside Lightning's DDP training process.  A
checkpoint is copied first, so sampling observes one complete epoch while the
trainer may safely advance to the next one.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
from omegaconf import OmegaConf

from ifmdock.metrics.docking import compute_ligand_rmsd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path,
        help="Model configuration file; defaults to RUN_DIR/model_parameters.yml.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/posebustersv1")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-complex", type=int, default=40)
    parser.add_argument("--inference-steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPUs for sharded parallel sampling")
    parser.add_argument(
        "--limit-complexes",
        type=int,
        default=0,
        help="Use only the first N sorted eligible complexes (0 means all).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-update-best-model", action="store_false", dest="update_best_model",
        help="Do not write a new best checkpoint into the source model directory.",
    )
    parser.add_argument(
        "--resume-existing", action="store_true",
        help="Keep completed docking_predictions.pkl files and sample only missing complexes.",
    )
    parser.add_argument(
        "--distance-candidate-selection",
        choices=("model_default", "single", "random", "balanced_random", "oracle_best"),
        default="model_default",
        help="Override cached UniMol distance-candidate selection for sampling.",
    )
    parser.add_argument(
        "--rdkit-initial-pose", action=argparse.BooleanOptionalAction, default=True,
        help="Use a freshly RDKit-generated ligand conformer before sampling.",
    )
    parser.add_argument(
        "--rdkit-initial-pose-mode",
        choices=("single", "paired", "matched_paired"), default="paired",
    )
    parser.add_argument(
        "--rdkit-initial-pose-root", type=Path, default=None,
        help="Sidecar directory containing the exact RDKit coordinates used for cache generation.",
    )
    parser.add_argument(
        "--require-paired-distance-candidates", action="store_true",
        help="Only export complexes with exactly ten cached paired RDKit/UniMol candidates.",
    )
    parser.add_argument(
        "--continue-on-shard-failure", action=argparse.BooleanOptionalAction, default=True,
        help="Keep successful GPU shards and write failures instead of aborting the full evaluation.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export and score existing prediction PKLs without running sampling again.",
    )
    return parser.parse_args()


def make_complete_residue_pocket(complex_dir: Path, name: str) -> Path | None:
    """Return the canonical UniMol/EC-Dock pocket used by the distance model.

    Rigid-pocket featurisation supports cropped residues through its nearest-CA
    fallback.  Creating a second whole-residue pocket here would change the
    frozen UniMol model's input distribution and break train/test equivalence.
    """
    output = complex_dir / f"{name}_protein_256.pdb"
    if output.is_file() and output.stat().st_size > 80:
        return output
    return None


def write_inputs(data_dir: Path, csv_path: Path, limit_complexes: int = 0,
                 require_paired_distance_candidates: bool = False) -> int:
    # The PDBBind rebuild scans ~19k input SDFs.  Per-file 2D/3D RDKit
    # warnings would otherwise flood the progress stream and hide the bar.
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for complex_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        name = complex_dir.name
        ligand = complex_dir / f"{name}_ligand.sdf"
        pocket = make_complete_residue_pocket(complex_dir, name)
        paired_ready = True
        if require_paired_distance_candidates:
            cache = complex_dir / f"interaction_{name}_v2.pkl"
            try:
                with cache.open("rb") as handle:
                    candidates = pickle.load(handle).get("rdkit_candidate_coords_list", [])
                # Match IFMDock's ComplexParser exactly: it reads the SDF with
                # sanitize=False and then applies RemoveHs.  Some charged or
                # unusual ligands retain explicit H atoms under this path, so
                # using a separately sanitized RDKit molecule here would let
                # an incompatible UniMol cache slip through preflight.
                supplier = Chem.SDMolSupplier(str(ligand), sanitize=False, removeHs=False)
                source = supplier[0] if len(supplier) else None
                expected_atoms = (
                    Chem.RemoveHs(Chem.Mol(source), sanitize=False).GetNumAtoms()
                    if source is not None else -1
                )
                paired_ready = (
                    len(candidates) == 10
                    and expected_atoms > 0
                    and all(np.asarray(coords).shape == (expected_atoms, 3) for coords in candidates)
                )
            except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
                paired_ready = False
        if ligand.is_file() and pocket is not None and paired_ready:
            rows.append(
                {
                    "pdbid": name,
                    "base_dir": str(complex_dir),
                    "apo_protein_file": str(pocket),
                    "ligand_input": str(ligand),
                    "ligand_description": "filename",
                    "ligand_path": str(ligand),
                }
            )
            if limit_complexes and len(rows) >= limit_complexes:
                break
    if not rows:
        raise RuntimeError("No PoseBusters complexes with ligand SDF and protein_256 PDB")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return len(rows)


def load_reference(path: Path):
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        return None, None
    mol = Chem.RemoveHs(mol)
    return mol, np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)


def export_predictions_to_sdf(data_dir: Path, prediction_dir: Path, inputs: pd.DataFrame):
    """Write all sampled ligand coordinates as one multi-record SDF per complex."""
    export_status = {}
    for row in inputs.itertuples(index=False):
        name = row.pdbid
        output_dir = prediction_dir / name
        prediction_path = output_dir / "docking_predictions.pkl"
        try:
            if not prediction_path.is_file():
                raise FileNotFoundError("missing_predictions")
            with prediction_path.open("rb") as handle:
                prediction = pickle.load(handle)
            source = Chem.SDMolSupplier(
                str(data_dir / name / f"{name}_ligand.sdf"),
                sanitize=True,
                removeHs=False,
            )
            source_mol = source[0] if len(source) else None
            if source_mol is None:
                raise ValueError("invalid_reference_ligand")
            predicted = [np.asarray(coords, dtype=np.float64) for coords in prediction["ligand_pos"]]
            if not predicted:
                raise ValueError("empty_predictions")
            n_atoms = predicted[0].shape[0]
            if source_mol.GetNumAtoms() == n_atoms:
                template = source_mol
            else:
                heavy_template = Chem.RemoveHs(source_mol)
                if heavy_template.GetNumAtoms() != n_atoms:
                    raise ValueError(
                        f"atom_count_mismatch prediction={n_atoms}, "
                        f"reference={source_mol.GetNumAtoms()}, heavy_reference={heavy_template.GetNumAtoms()}"
                    )
                template = heavy_template
            sdf_path = output_dir / f"gen_{name}_ligand.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            for sample_index, coords in enumerate(predicted):
                if coords.shape != (template.GetNumAtoms(), 3):
                    raise ValueError(f"sample_{sample_index}_shape={coords.shape}")
                mol = Chem.Mol(template)
                conformer = mol.GetConformer()
                for atom_index, xyz in enumerate(coords):
                    conformer.SetAtomPosition(atom_index, Point3D(*map(float, xyz)))
                mol.SetProp("sample_index", str(sample_index))
                mol.SetProp("source_checkpoint_epoch", str(prediction.get("epoch", "unknown")))
                writer.write(mol)
            writer.close()
            export_status[name] = {"status": "ok", "sdf_file": str(sdf_path), "num_samples": len(predicted)}
        except Exception as exc:
            export_status[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return export_status


def score_predictions(data_dir: Path, prediction_dir: Path, inputs: pd.DataFrame):
    per_complex = []
    for row in inputs.itertuples(index=False):
        name = row.pdbid
        output_file = prediction_dir / name / "docking_predictions.pkl"
        record = {"complex_id": name, "status": "failed"}
        try:
            if not output_file.is_file():
                record["reason"] = "missing_predictions"
                per_complex.append(record)
                continue
            mol, ref_coords = load_reference(data_dir / name / f"{name}_ligand.sdf")
            if mol is None:
                record["reason"] = "invalid_reference_ligand"
                per_complex.append(record)
                continue
            with output_file.open("rb") as handle:
                prediction = pickle.load(handle)
            heavy_mask = np.asarray(prediction["filterHs"], dtype=bool)
            rmsds = []
            for coords in prediction["ligand_pos"]:
                coords = np.asarray(coords, dtype=np.float64)
                # Prediction payloads from different IFMDock versions store
                # either all-atom coordinates plus filterHs or coordinates
                # already reduced to heavy atoms. Never apply the mask twice.
                if coords.shape[0] == ref_coords.shape[0]:
                    pass
                elif coords.shape[0] == heavy_mask.shape[0]:
                    coords = coords[heavy_mask]
                else:
                    raise ValueError(
                        f"cannot align predicted atoms={coords.shape[0]}, "
                        f"mask={heavy_mask.shape[0]}, reference={ref_coords.shape[0]}"
                    )
                if coords.shape != ref_coords.shape:
                    raise ValueError(f"shape mismatch predicted={coords.shape}, reference={ref_coords.shape}")
                rmsds.append(float(compute_ligand_rmsd(ref_coords, coords, name, mol)))
            if not rmsds:
                record["reason"] = "empty_predictions"
                per_complex.append(record)
                continue
            rmsds = np.asarray(rmsds, dtype=np.float64)
            record.update(
                status="ok",
                num_samples=int(len(rmsds)),
                best_rmsd=float(rmsds.min()),
                mean_rmsd=float(rmsds.mean()),
                best1_pass=bool(rmsds.min() <= 2.0),
                average_pass=bool(rmsds.mean() <= 2.0),
                rmsds=rmsds.round(6).tolist(),
            )
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        per_complex.append(record)
    successful = [row for row in per_complex if row["status"] == "ok"]
    return per_complex, successful


def update_summary(path: Path, metrics: dict):
    previous = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("epoch") != metrics["epoch"]:
                    previous.append(row)
            except json.JSONDecodeError:
                continue
    previous.append(metrics)
    previous.sort(key=lambda row: row["epoch"])
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in previous))
    os.replace(tmp, path)


def update_best_model(run_dir: Path, snapshot_checkpoint: Path, metrics: dict):
    """Keep the checkpoint with the highest complex-level mean-RMSD pass rate."""
    metadata_path = run_dir / "best_posebusters_average_metrics.json"
    try:
        previous = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        previous = {}
    current_rate = metrics.get("average_rate")
    previous_rate = previous.get("average_rate")
    if current_rate is None or (previous_rate is not None and current_rate <= previous_rate):
        return False
    target = run_dir / "best_posebusters_average_model.pt"
    temporary = target.with_suffix(".pt.tmp")
    shutil.copy2(snapshot_checkpoint, temporary)
    os.replace(temporary, target)
    metadata = {**metrics, "checkpoint": str(target)}
    tmp_metadata = metadata_path.with_suffix(".json.tmp")
    tmp_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(tmp_metadata, metadata_path)
    return True


def main():
    args = parse_args()
    epoch_dir = args.output_root / f"epoch_{args.epoch:04d}"
    metrics_path = epoch_dir / "metrics.json"
    if metrics_path.is_file() and not args.force:
        print(f"Already evaluated: {metrics_path}")
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    inputs_csv = epoch_dir / "inputs.csv"
    n_requested = write_inputs(
        args.data_dir, inputs_csv, args.limit_complexes,
        require_paired_distance_candidates=args.require_paired_distance_candidates,
    )

    prediction_dir = epoch_dir / "predictions"
    started_at = time.time()
    inputs = pd.read_csv(inputs_csv)
    if not args.export_only:
        # ``last_model.pt`` is atomically refreshed by training.  Freeze an
        # epoch snapshot before sampling so the result remains reproducible.
        snapshot_dir = epoch_dir / "model"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot_checkpoint = snapshot_dir / "checkpoint.pt"
        shutil.copy2(args.checkpoint, snapshot_checkpoint)
        model_config = args.model_config or args.run_dir / "model_parameters.yml"
        shutil.copy2(model_config, snapshot_dir / "model_parameters.yml")
        if args.distance_candidate_selection != "model_default":
            snapshot_config = OmegaConf.load(snapshot_dir / "model_parameters.yml")
            snapshot_config.model.distance_guidance.cached_candidate_selection = (
                args.distance_candidate_selection
            )
            OmegaConf.save(snapshot_config, snapshot_dir / "model_parameters.yml")
        sampling_inputs = inputs
        if args.resume_existing:
            pending_mask = [
                not (prediction_dir / str(row.pdbid) / "docking_predictions.pkl").is_file()
                for row in inputs.itertuples(index=False)
            ]
            sampling_inputs = inputs.loc[pending_mask].reset_index(drop=True)
            print(
                f"Resuming with {len(sampling_inputs)} pending complexes; "
                f"keeping {len(inputs) - len(sampling_inputs)} completed complexes",
                flush=True,
            )
        gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
        shards = gpu_ids or [None]
        processes = []
        for shard_index, gpu in enumerate(shards):
            shard_inputs = sampling_inputs.iloc[shard_index::len(shards)]
            if shard_inputs.empty:
                continue
            shard_csv = epoch_dir / f"inputs_shard_{shard_index}.csv"
            shard_inputs.to_csv(shard_csv, index=False)
            command = [
                sys.executable, str(Path(__file__).with_name("predict.py")),
                "--input_csv", str(shard_csv),
                "--output_dir", str(prediction_dir),
                "--docking_model_dir", str(snapshot_dir),
                "--docking_ckpt", "checkpoint.pt",
                "--samples_per_complex", str(args.samples_per_complex),
                "--inference_steps", str(args.inference_steps),
                "--batch_size", str(args.batch_size),
                "--rigid_pocket", "--use_fast_sampling",
                "--failure-log", str(epoch_dir / f"predict_gpu_{gpu or 'default'}_invalid.jsonl"),
            ]
            if args.rdkit_initial_pose:
                command.append("--rdkit_initial_pose")
                command.extend(["--rdkit_initial_pose_mode", args.rdkit_initial_pose_mode])
            else:
                # ``predict.py`` enables RDKit initialisation by default.
                # Omitting the positive flag therefore does *not* disable it;
                # explicitly propagate the negative BooleanOptionalAction.
                command.append("--no-rdkit_initial_pose")
            if args.rdkit_initial_pose_root is not None:
                command.extend(["--rdkit_initial_pose_root", str(args.rdkit_initial_pose_root)])
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            log = (epoch_dir / f"predict_gpu_{gpu or 'default'}.log").open("w")
            processes.append((gpu, log, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)))
        failures = []
        total = len(inputs)
        input_names = set(inputs["pdbid"].astype(str))
        active = list(processes)
        while active:
            completed = sum(
                (prediction_dir / name / "docking_predictions.pkl").is_file()
                for name in input_names
            )
            width = 30
            filled = min(width, (completed * width // total) if total else width)
            print(
                f"Stage-1 sampling [{'#' * filled}{'.' * (width - filled)}] "
                f"{completed}/{total} ({(100.0 * completed / total) if total else 100.0:.1f}%) "
                f"active_shards={len(active)}",
                flush=True,
            )
            time.sleep(15)
            still_active = []
            for gpu, log, process in active:
                returncode = process.poll()
                if returncode is None:
                    still_active.append((gpu, log, process))
                    continue
                log.close()
                if returncode:
                    failures.append(f"GPU {gpu}: exit {returncode}")
            active = still_active
        if failures:
            (epoch_dir / "sampling_shard_failures.json").write_text(
                json.dumps({"failures": failures, "action": "continued_with_completed_shards"}, indent=2) + "\n"
            )
            if not args.continue_on_shard_failure:
                raise RuntimeError("Parallel sampling failed: " + ", ".join(failures))
            print("WARNING: " + "; ".join(failures) + "; retaining completed shards", flush=True)
    export_status = export_predictions_to_sdf(args.data_dir, prediction_dir, inputs)
    (epoch_dir / "sdf_export.json").write_text(json.dumps(export_status, indent=2) + "\n")
    per_complex, successful = score_predictions(args.data_dir, prediction_dir, inputs)
    best1 = sum(row["best1_pass"] for row in successful)
    average = sum(row["average_pass"] for row in successful)
    metrics = {
        "epoch": args.epoch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_complexes": n_requested,
        "evaluated_complexes": len(successful),
        "failed_complexes": n_requested - len(successful),
        "samples_per_complex": args.samples_per_complex,
        "inference_steps": args.inference_steps,
        "best1_pass_count": best1,
        "best1_rate": best1 / len(successful) if successful else None,
        "average_pass_count": average,
        "average_rate": average / len(successful) if successful else None,
        "mean_best_rmsd": float(np.mean([row["best_rmsd"] for row in successful])) if successful else None,
        "mean_rmsd": float(np.mean([row["mean_rmsd"] for row in successful])) if successful else None,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    (epoch_dir / "per_complex_rmsd.json").write_text(json.dumps(per_complex, indent=2) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    update_summary(args.output_root / "posebusters_metrics.jsonl", metrics)
    metrics["saved_as_best_average_model"] = (
        update_best_model(args.run_dir, epoch_dir / "model" / "checkpoint.pt", metrics)
        if args.update_best_model else False
    )
    # Rewrite after adding the best-model decision, and provide a convenient
    # tabular history in addition to the durable JSONL record.
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    update_summary(args.output_root / "posebusters_metrics.jsonl", metrics)
    summary_rows = [json.loads(line) for line in
                    (args.output_root / "posebusters_metrics.jsonl").read_text().splitlines()
                    if line.strip()]
    pd.DataFrame(summary_rows).to_csv(args.output_root / "posebusters_metrics.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
