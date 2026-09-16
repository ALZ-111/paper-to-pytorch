"""python -m pytest utils/test_utils.py -v   (from the repo root)"""

import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import count_parameters, get_device, seed_everything  # noqa: E402
from utils.results import load_results, update_results  # noqa: E402


def test_get_device_returns_a_usable_device():
    d = get_device()
    assert torch.zeros(1, device=d).device.type == d.type


def test_seed_everything_makes_torch_and_random_deterministic():
    import random
    seed_everything(123)
    a = (torch.randn(3).tolist(), random.random())
    seed_everything(123)
    b = (torch.randn(3).tolist(), random.random())
    assert a == b


def test_count_parameters_counts_tied_weights_once():
    emb = nn.Embedding(10, 4)
    out = nn.Linear(4, 10)
    out.weight = emb.weight  # tie
    model = nn.Sequential(emb, out)
    assert count_parameters(model) == 10 * 4 + 10  # embedding + output bias only
    emb.weight.requires_grad_(False)
    assert count_parameters(model) == 10
    assert count_parameters(model, trainable_only=False) == 50


def test_update_results_merges_namespaces_and_survives_missing_file(tmp_path):
    path = str(tmp_path / "r.json")
    assert load_results(path) == {}
    update_results(path, {"a": 1})
    update_results(path, {"bleu": 40.0}, namespace="bpe")
    update_results(path, {"ppl": 5.0}, namespace="bpe")
    update_results(path, {"a": 2})
    data = json.load(open(path))
    assert data == {"a": 2, "bpe": {"bleu": 40.0, "ppl": 5.0}}
    assert not os.path.exists(path + ".tmp")
