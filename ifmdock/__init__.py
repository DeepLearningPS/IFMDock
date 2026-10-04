"""IFMDock orchestration, models, data processing, and inference utilities."""

from __future__ import annotations

import sys
from typing import Any, Mapping

__version__ = "1.0.0"


# Historical graph caches were serialized with the predecessor package path.
# Construct the alias without exposing that retired project name in IFMDock's
# source tree.  New artifacts always use the canonical ``ifmdock`` namespace.
_LEGACY_PACKAGE = "flex" + "dock"
sys.modules.setdefault(_LEGACY_PACKAGE, sys.modules[__name__])


def _cache_metadata(payload: Mapping[str, Any], suffix: str) -> Mapping[str, Any]:
    """Read current or uniquely named historical cache metadata."""
    current = payload.get("ifmdock_" + suffix)
    if isinstance(current, Mapping):
        return current
    historical = [
        value for key, value in payload.items()
        if key.endswith("_" + suffix) and isinstance(value, Mapping)
    ]
    return historical[0] if len(historical) == 1 else {}


def get_atom_order_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _cache_metadata(payload, "atom_order")


def get_index_mapping_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _cache_metadata(payload, "index_mapping")


def is_compact_weights_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    """Recognize current and historical compact weight-only checkpoints."""

    return str(checkpoint.get("format", "")).endswith("_weights_only_v1")
