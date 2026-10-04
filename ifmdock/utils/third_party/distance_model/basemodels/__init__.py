"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]
from Distance_model.basemodels.distributed import utils as distributed_utils
from Distance_model.basemodels.logging import meters, metrics, progress_bar

sys.modules["Distance_model.basemodels.distributed_utils"] = distributed_utils
sys.modules["Distance_model.basemodels.meters"] = meters
sys.modules["Distance_model.basemodels.metrics"] = metrics
sys.modules["Distance_model.basemodels.progress_bar"] = progress_bar

import Distance_model.basemodels.losses
import Distance_model.basemodels.distributed
import Distance_model.basemodels.models
import Distance_model.basemodels.modules
import Distance_model.basemodels.optim
import Distance_model.basemodels.optim.lr_scheduler
import Distance_model.basemodels.tasks

