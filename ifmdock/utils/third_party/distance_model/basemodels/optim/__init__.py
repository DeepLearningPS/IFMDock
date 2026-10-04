"""isort:skip_file"""

import importlib
import os

from Distance_model.basemodels import registry
from Distance_model.basemodels.optim.optimizer import (
    UnicoreOptimizer,
)
from Distance_model.basemodels.optim.fp16_optimizer import FP16Optimizer

__all__ = [
    "UnicoreOptimizer",
    "FP16Optimizer",
]

(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY
) = registry.setup_registry("--optimizer", base_class=UnicoreOptimizer, default='adam')


def build_optimizer(args, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(args, params, *extra_args, **extra_kwargs)
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("Distance_model.basemodels.optim." + file_name)
