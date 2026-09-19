"""Tests for checkpoint averaging.  python -m pytest test_average.py -v"""

import torch

import model as m
import _bootstrap  # noqa: F401
from utils.checkpoint import load_checkpoint
from average_checkpoints import average_checkpoints, average_state_dicts


def _small(seed):
    torch.manual_seed(seed)
    return m.Transformer(30, 30, d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0)


def test_average_is_elementwise_mean():
    a, b = _small(0), _small(1)
    avg = average_state_dicts([a.state_dict(), b.state_dict()])
    for k in avg:
        expected = (a.state_dict()[k].float() + b.state_dict()[k].float()) / 2
        assert torch.allclose(avg[k].float(), expected)


def test_average_of_identical_checkpoints_is_identity_and_loads(tmp_path):
    net = _small(0)
    src = torch.randint(1, 30, (2, 5))
    tgt = torch.randint(1, 30, (2, 4))
    ref = net(src, tgt)
    paths = []
    for e in (1, 2, 3):
        p = tmp_path / f"epoch{e}.pt"
        torch.save({"model": net.state_dict(), "args": {}, "epoch": e}, p)
        paths.append(str(p))
    # half=False isolates the averaging arithmetic, which is exact for identical inputs,
    # from checkpoint storage precision (covered by utils/test_utils.py).
    merged = average_checkpoints(paths, str(tmp_path / "avg.pt"), half=False)
    assert merged["averaged_from"] == [1, 2, 3]
    net.eval()
    net2 = _small(5)
    net2.load_state_dict(load_checkpoint(str(tmp_path / "avg.pt"))["model"])
    net2.eval()
    assert torch.allclose(net2(src, tgt), ref, atol=1e-6)

    # The default float16 storage is lossy but far inside what the model cares about.
    average_checkpoints(paths, str(tmp_path / "avg16.pt"))
    net3 = _small(5)
    net3.load_state_dict(load_checkpoint(str(tmp_path / "avg16.pt"))["model"])
    net3.eval()
    assert torch.allclose(net3(src, tgt), ref, atol=2e-2)


def test_tied_weights_survive_averaging():
    """generator.weight is the same tensor as decoder.embed.weight when tied; the
    averaged state dict must keep them equal so load_state_dict stays consistent."""
    torch.manual_seed(0)
    nets = [m.Transformer(30, 30, d_model=16, n_layers=1, n_heads=2, d_ff=32, tie_weights=True)
            for _ in range(3)]
    avg = average_state_dicts([n.state_dict() for n in nets])
    assert torch.equal(avg["generator.weight"], avg["decoder.embed.weight"])
