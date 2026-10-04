#!/usr/bin/env python3
"""Regenerate canonical UniMol/EC-Dock pockets from existing docking grids."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biopandas.pdb import PandasPdb

from ifmdock.preprocessing.unimol_pocket import (
    atom_signature,
    generate_unimol_pocket,
    read_docking_grid,
    write_unimol_pocket,
)


def generate_one(task):
    directory, max_atoms, overwrite = task
    directory = Path(directory)
    name = directory.name
    protein = directory / f"{name}_protein.pdb"
    grid_path = directory / f"{name}_ligand_docking_grid_boxsize10.json"
    output = directory / f"{name}_protein_{max_atoms}.pdb"
    if not protein.is_file() or not grid_path.is_file():
        return name, "failed", 0, 0, "missing protein.pdb or docking grid"
    try:
        selected, metadata = generate_unimol_pocket(
            protein, read_docking_grid(grid_path), max_atoms
        )
        previous = None
        if output.is_file():
            previous = PandasPdb().read_pdb(str(output)).df["ATOM"]
        changed = previous is None or atom_signature(previous) != atom_signature(selected)
        if overwrite:
            write_unimol_pocket(selected, output)
        return (
            name, "changed" if changed else "unchanged",
            metadata["selected_atoms"], metadata["box_atoms"], "",
        )
    except Exception as exc:
        return name, "failed", 0, 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--max-pocket-atoms", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    directories = []
    for root in args.roots:
        directories.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    if args.limit:
        directories = directories[:args.limit]
    tasks = ((str(path), args.max_pocket_atoms, args.overwrite) for path in directories)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(generate_one, tasks, chunksize=8), 1):
            rows.append(row)
            if index % 500 == 0:
                print(f"processed={index}/{len(directories)}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["complex", "status", "selected_atoms", "box_atoms", "detail"])
        writer.writerows(rows)
    os.replace(temporary, args.report)
    counts = {status: sum(row[1] == status for row in rows)
              for status in ("changed", "unchanged", "failed")}
    print(json.dumps({"total": len(rows), **counts, "report": str(args.report)}, indent=2))
    if counts["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
