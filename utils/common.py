import random

import torch


def get_device():
    """CUDA if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    """Seed Python, NumPy (if installed) and torch (CPU and CUDA)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model, trainable_only=True):
    """Number of (trainable) parameters. Tied weights are counted once, because
    model.parameters() de-duplicates shared tensors."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
