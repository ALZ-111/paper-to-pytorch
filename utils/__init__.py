"""
Shared helpers for every paper folder.

    from utils import get_device, seed_everything, count_parameters
    from utils.results import load_results, update_results
    from utils.plotting import plt, save_fig

Paper scripts are run from their own folder (`python train.py`), so the repo root is
not on sys.path by default. Each paper folder has a two-line `_bootstrap.py` that adds
it; scripts import that first.
"""

from .common import count_parameters, get_device, seed_everything

__all__ = ["count_parameters", "get_device", "seed_everything"]
