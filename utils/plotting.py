"""
Headless matplotlib with the import-order fix every figure script needs.

Anaconda ships two OpenMP runtimes (torch's libiomp5 and matplotlib's libomp). If
matplotlib is imported first the process aborts when torch loads, so this module
imports torch before matplotlib and selects the Agg backend. Import `plt` from here
instead of from matplotlib.
"""

import os

import torch  # noqa: F401  (must precede matplotlib, see module docstring)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt", "save_fig"]


def save_fig(fig, path, dpi=130, tight=True):
    """Save and close a figure, creating the directory if needed. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
