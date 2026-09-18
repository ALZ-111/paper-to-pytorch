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


def test_checkpoint_half_precision_roundtrip(tmp_path):
    from utils.checkpoint import load_checkpoint, save_checkpoint
    net = nn.Sequential(nn.Linear(64, 64), nn.Embedding(10, 64))
    state = {"model": net.state_dict(), "epoch": 7, "itos": ["a", "b"],
             "counts": torch.arange(5)}  # non-float tensors must survive untouched
    half = save_checkpoint(state, str(tmp_path / "h.pt"))
    full = save_checkpoint(state, str(tmp_path / "f.pt"), half=False)
    assert os.path.getsize(half) < 0.6 * os.path.getsize(full)

    back = load_checkpoint(half)
    assert back["epoch"] == 7 and back["itos"] == ["a", "b"]
    assert torch.equal(back["counts"], state["counts"])
    assert "_storage_dtype" not in back
    for k, v in back["model"].items():
        assert v.dtype == torch.float32
        assert torch.allclose(v, state["model"][k], atol=1e-2, rtol=1e-2)
    net.load_state_dict(back["model"])  # loads without dtype complaints

    # A checkpoint written the old way (plain torch.save, float32) still loads.
    torch.save(state, tmp_path / "legacy.pt")
    old = load_checkpoint(str(tmp_path / "legacy.pt"))
    assert torch.equal(old["model"]["0.weight"], state["model"]["0.weight"])
