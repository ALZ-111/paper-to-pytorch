"""Model and metric tests.  python -m pytest test_model.py -v"""

import math

import torch
import torch.nn as nn

import model as M
from metrics import evaluate, hit_ratio, ndcg, positive_rank

U, I, B = 50, 80, 32


def _batch():
    torch.manual_seed(0)
    return torch.randint(0, U, (B,)), torch.randint(0, I, (B,))


def test_all_models_return_one_logit_per_pair():
    u, i = _batch()
    for kind in ("gmf", "mlp", "neumf"):
        out = M.build(kind, U, I, factors=8)(u, i)
        assert out.shape == (B,) and torch.isfinite(out).all()


def test_gmf_with_unit_weights_is_matrix_factorisation():
    """Eq. 9 generalises MF: h = 1 and no bias gives exactly the dot product p_u . q_i."""
    g = M.GMF(U, I, 8)
    with torch.no_grad():
        g.out.weight.fill_(1.0)
        g.out.bias.zero_()
    u, i = _batch()
    assert torch.allclose(g(u, i), (g.user(u) * g.item(i)).sum(-1), atol=1e-6)


def test_mlp_tower_matches_the_authors_architecture():
    m = M.build("mlp", U, I, factors=8)
    assert m.user.embedding_dim == m.item.embedding_dim == 32
    widths = [(l.in_features, l.out_features) for l in m.tower if isinstance(l, nn.Linear)]
    assert widths == [(64, 32), (32, 16), (16, 8)]            # three hidden layers
    assert m.out.in_features == 8


def test_neumf_has_separate_branch_embeddings_and_no_dead_parameters():
    n = M.build("neumf", U, I, factors=8)
    assert n.gmf.user.embedding_dim == 8 and n.mlp.user.embedding_dim == 32
    assert n.out.in_features == 8 + 8
    names = [k for k, _ in n.named_parameters()]
    assert not any(k.startswith(("gmf.out", "mlp.out")) for k in names)
    # every parameter receives a gradient
    u, i = _batch()
    n(u, i).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in n.parameters())


def test_pretrained_neumf_starts_as_the_alpha_blend_of_gmf_and_mlp():
    """With h = [a*h_G ; (1-a)*h_M] and the biases blended the same way, NeuMF's logit
    before any training is exactly a*GMF + (1-a)*MLP."""
    torch.manual_seed(1)
    g, m = M.build("gmf", U, I, 8), M.build("mlp", U, I, 8)
    with torch.no_grad():                           # make both branches non-trivial
        g.out.bias.fill_(0.3); m.out.bias.fill_(-0.7)
    u, i = _batch()
    for alpha in (0.5, 0.2):
        n = M.build("neumf", U, I, 8).load_pretrained(g, m, alpha=alpha)
        expected = alpha * g(u, i) + (1 - alpha) * m(u, i)
        assert torch.allclose(n(u, i), expected, atol=1e-6)


def test_metrics_on_hand_computed_cases():
    scores = torch.tensor([[5.0, 1.0, 2.0, 3.0],    # positive first -> rank 0
                           [2.0, 3.0, 4.0, 1.0],    # two negatives above -> rank 2
                           [1.0, 2.0, 3.0, 4.0]])   # last -> rank 3
    rank = positive_rank(scores)
    assert rank.tolist() == [0, 2, 3]
    assert math.isclose(hit_ratio(rank, k=3), 2 / 3, rel_tol=1e-6)
    assert math.isclose(ndcg(rank, k=3), (1 + 1 / math.log2(4)) / 3, rel_tol=1e-6)


def test_ties_count_against_the_positive():
    """A model that scores every item equally must not get a perfect hit ratio."""
    rank = positive_rank(torch.zeros(10, 100))
    assert hit_ratio(rank) == 0.0 and ndcg(rank) == 0.0


def test_evaluate_gives_chance_level_for_a_random_model_and_1_for_an_oracle():
    torch.manual_seed(0)
    n_users, n_items, C = 3000, 500, 100          # need at least C distinct items per row
    cand = torch.stack([torch.randperm(n_items)[:C] for _ in range(n_users)])
    assert cand.shape == (n_users, C)

    class Random(nn.Module):
        def forward(self, u, i):
            return torch.rand(u.shape)

    hr, nd = evaluate(Random(), cand)
    chance_ndcg = sum(1 / math.log2(r + 2) for r in range(10)) / C
    assert abs(hr - 0.10) < 0.02 and abs(nd - chance_ndcg) < 0.01

    pos = cand[:, 0].clone()

    class Oracle(nn.Module):
        def forward(self, u, i):
            return (i == pos[u]).float()

    assert evaluate(Oracle(), cand) == (1.0, 1.0)
