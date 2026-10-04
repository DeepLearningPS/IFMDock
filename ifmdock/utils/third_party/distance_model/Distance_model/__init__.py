"""Compatibility package for the original UniMol Docking V2 import name.

The upstream code imports ``Distance_model.*`` while IFMDock vendors that tree
under ``ifmdock/utils/third_party/distance_model``. Resolve that import from the same
directory without duplicating the model code.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1])]
