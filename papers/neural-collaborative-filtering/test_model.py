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


def test_sparse_embeddings_do_not_change_the_forward_pass():
    torch.manual_seed(0)
    dense = M.build("neumf", U, I, 8)
    sparse = M.build("neumf", U, I, 8, sparse=True)
    sparse.load_state_dict(dense.state_dict())
    u, i = _batch()
    assert torch.allclose(dense(u, i), sparse(u, i), atol=1e-6)
    emb, rest = M.split_parameters(sparse)
    assert len(emb) == 4                                # two tables per branch
    assert M.split_parameters(dense)[0] == []           # nothing sparse to split out
    # At MovieLens sizes the tables are the overwhelming majority of the parameters,
    # which is why skipping the untouched rows is worth doing at all: 99% at 8 factors,
    # still 94% at 64 (the tower grows quadratically with factors, the tables linearly).
    for factors, share in ((8, 0.98), (64, 0.9)):
        b_emb, b_rest = M.split_parameters(M.build("mlp", 6040, 3706, factors, sparse=True))
        n_emb = sum(p.numel() for p in b_emb)
        assert n_emb / (n_emb + sum(p.numel() for p in b_rest)) > share


def test_sparse_and_dense_adam_agree_on_the_first_step_then_diverge_on_untouched_rows():
    """Two differences, both worth pinning down.

    Step 1: the updates agree only up to epsilon handling. Dense Adam divides the second
    moment by its bias correction and *then* adds eps; SparseAdam adds eps to the raw
    sqrt and folds the correction into the step size. At step 1 that scales eps by
    sqrt(1 - beta2) = 0.032, so the two differ by ~1e-3 with the default eps=1e-8 and
    agree to 1e-8 when eps is negligible.

    Step 2 onward: dense Adam keeps moving rows that received no gradient, because their
    momentum is still non-zero, while SparseAdam touches only the rows in the batch. That
    is the real semantic price of the speed-up."""
    def setup(sparse, eps=1e-8):
        torch.manual_seed(0)
        m = M.build("gmf", U, I, 8, sparse=sparse)
        if sparse:
            emb, rest = M.split_parameters(m)
            return m, [torch.optim.SparseAdam(emb, lr=0.1, eps=eps),
                       torch.optim.Adam(rest, lr=0.1, eps=eps)]
        return m, [torch.optim.Adam(m.parameters(), lr=0.1, eps=eps)]

    def step(m, opts, u, i):
        loss = nn.functional.binary_cross_entropy_with_logits(m(u, i), torch.ones(len(u)))
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()

    first_u, first_i = torch.tensor([0, 1]), torch.tensor([0, 1])
    second_u, second_i = torch.tensor([2, 3]), torch.tensor([2, 3])

    # With eps negligible the two optimizers' first step is the same computation.
    tiny = [setup(s, eps=1e-16) for s in (False, True)]
    for m, opts in tiny:
        step(m, opts, first_u, first_i)
    assert torch.allclose(tiny[0][0].user.weight, tiny[1][0].user.weight, atol=1e-6)

    md, od = setup(False)
    ms, os_ = setup(True)
    step(md, od, first_u, first_i)
    step(ms, os_, first_u, first_i)
    # With the default eps they differ, but only slightly, and only on touched rows.
    delta = (md.user.weight - ms.user.weight).abs().max()
    assert 0 < delta < 1e-2

    before_d = md.user.weight[0].clone()
    before_s = ms.user.weight[0].clone()
    step(md, od, second_u, second_i)      # user 0 gets no gradient this step
    step(ms, os_, second_u, second_i)
    assert not torch.allclose(md.user.weight[0], before_d)             # dense keeps moving it
    assert torch.equal(ms.user.weight[0], before_s)                    # sparse leaves it alone


def test_l2_penalty_covers_only_the_rows_a_batch_uses():
    torch.manual_seed(0)
    for kind in ("gmf", "mlp", "neumf"):
        m = M.build(kind, U, I, 8)
        u, i = torch.tensor([3, 3]), torch.tensor([7, 7])          # one user, one item, twice
        pen = m.l2_penalty(u, i)
        tables = [(m.user, m.item)] if kind != "neumf" else [(m.gmf.user, m.gmf.item),
                                                             (m.mlp.user, m.mlp.item)]
        expected = sum(2 * (ut(u[:1]).pow(2).sum() + it(i[:1]).pow(2).sum()) for ut, it in tables)
        assert torch.allclose(pen, expected, atol=1e-6)             # repeats counted twice
        # a row outside the batch contributes nothing
        with torch.no_grad():
            for ut, _ in tables:
                ut.weight[5].fill_(100.0)
        assert torch.allclose(m.l2_penalty(u, i), pen, atol=1e-4)


def test_l2_penalty_shrinks_the_embeddings_it_penalises():
    torch.manual_seed(0)
    m = M.build("gmf", U, I, 8)
    with torch.no_grad():
        m.user.weight.fill_(0.5)
        m.item.weight.fill_(0.5)
    # lr * 2 * reg must stay well below 1, or the row overshoots and oscillates about 0
    # instead of decaying towards it.
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    u, i = torch.tensor([1]), torch.tensor([2])
    for _ in range(5):
        loss = 1.0 * m.l2_penalty(u, i)           # penalty alone, no data term
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert 0 < m.user.weight[1].abs().max() < 0.45    # penalised row shrank towards zero
    assert torch.allclose(m.user.weight[0], torch.full((8,), 0.5))   # untouched row did not
