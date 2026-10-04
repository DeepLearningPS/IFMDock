#!/usr/bin/env python3
"""Validate the runtime and the two released IFMDock checkpoints."""

from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    failed = []
    for module in ("torch", "torch_geometric", "rdkit", "omegaconf", "lightning", "posebusters"):
        try:
            loaded = importlib.import_module(module)
            print(f"OK  {module:18s} {getattr(loaded, '__version__', '')}")
        except Exception as exc:
            failed.append(f"{module}: {exc}")
    import torch
    for relative in ("checkpoints/model1.pt", "checkpoints/model2.pt",
                     "ifmdock/utils/third_party/distance_model/premodel/best.pt",
                     "IFMScore/trained_models/ifmscore.pth"):
        path = ROOT / relative
        if not path.is_file():
            failed.append(f"missing {path}")
            continue
        if relative.startswith("checkpoints"):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            print(f"OK  {relative} epoch={payload.get('epoch', 'unknown')}")
        else:
            print(f"OK  {relative} ({path.stat().st_size / 2**20:.1f} MiB)")
    if failed:
        print("\nFAILED\n" + "\n".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
