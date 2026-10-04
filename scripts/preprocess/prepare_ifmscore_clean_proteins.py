#!/usr/bin/env python3
"""Create cleaned full receptors for IFMScore scoring."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ifmdock.preprocessing.unimol_pocket import clean_full_protein, write_unimol_pocket


def process(directory: Path):
    name = directory.name
    source = directory / f"{name}_protein.pdb"
    target = directory / f"{name}_protein_ifmdock_clean.pdb"
    row = {"complex": name, "status": "failed", "input_atoms": 0,
           "output_atoms": 0, "output": str(target), "detail": ""}
    try:
        frame, metadata = clean_full_protein(source)
        write_unimol_pocket(frame, target)
        row.update(status="ok", input_atoms=metadata["input_atom_records"],
                   output_atoms=metadata["selected_atoms"])
    except Exception as exc:
        row["detail"] = f"{type(exc).__name__}: {exc}"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(path for path in args.dataset.iterdir() if path.is_dir())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, directories, chunksize=8))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fields = ["complex", "status", "input_atoms", "output_atoms", "output", "detail"]
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, args.report)
    failures = sum(row["status"] != "ok" for row in rows)
    print(f"processed={len(rows)} ok={len(rows)-failures} failed={failures}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
