"""
Data pipeline tests.  python -m pytest test_data.py -v

The first group runs on small synthetic data, so it needs no download and runs in CI.
The last group checks the real MovieLens-1M split and is skipped if it has not been
built yet (python data.py builds it).
"""

import os

import numpy as np
import pytest

import data as D


# ------------------------------------------------------------------ synthetic, CI-safe
def _toy():
    #            user item  time
    rows = [(10, 100, 5), (10, 101, 9), (10, 102, 7),     # user 10: latest is item 101
            (20, 100, 3), (20, 103, 3),                   # user 20: tie at t=3 -> later row wins
            (30, 104, 1), (30, 101, 2), (30, 100, 8)]     # user 30: latest is item 100
    a = np.array(rows, dtype=np.int64)
    return a[:, 0], a[:, 1], a[:, 2]


def test_reindex_is_dense_and_order_preserving():
    ids, n = D.reindex(np.array([50, 7, 50, 900, 7]))
    assert n == 3 and ids.tolist() == [1, 0, 1, 2, 0]


def test_leave_one_out_holds_out_each_users_latest_interaction():
    u, i, t = _toy()
    tr_u, tr_i, te_u, te_i = D.leave_one_out(u, i, t)
    assert te_u.tolist() == [10, 20, 30]
    assert te_i.tolist() == [101, 103, 100]          # tie broken by file order
    assert len(tr_u) + len(te_u) == len(u)
    # no held-out pair leaks into training
    train_pairs = set(zip(tr_u.tolist(), tr_i.tolist()))
    assert not any((a, b) in train_pairs for a, b in zip(te_u.tolist(), te_i.tolist()))


def test_test_negatives_are_unseen_distinct_and_deterministic():
    rng = np.random.default_rng(1)
    n_users, n_items = 40, 60
    users = np.repeat(np.arange(n_users), 15)
    items = np.concatenate([rng.choice(n_items, 15, replace=False) for _ in range(n_users)])
    test_u = np.arange(n_users)
    a = D.sample_test_negatives(users, items, test_u, n_items, num_neg=20, seed=7)
    b = D.sample_test_negatives(users, items, test_u, n_items, num_neg=20, seed=7)
    assert a.shape == (n_users, 20) and np.array_equal(a, b)
    for row, uu in enumerate(test_u):
        seen = set(items[users == uu].tolist())
        assert len(set(a[row].tolist())) == 20
        assert not seen & set(a[row].tolist())


def test_in_sorted_handles_edges():
    keys = np.array([3, 5, 9])
    q = np.array([0, 3, 4, 5, 9, 10])
    assert D.in_sorted(q, keys).tolist() == [False, True, False, True, True, False]


def test_negative_sampler_never_returns_a_training_item():
    rng = np.random.default_rng(0)
    n_items = 30
    # dense users (20 of 30 items) force many rejections and exercise the retry loop
    train_u = np.repeat(np.arange(8), 20)
    train_i = np.concatenate([rng.choice(n_items, 20, replace=False) for _ in range(8)])
    s = D.NegativeSampler(train_u, train_i, n_items)
    u, j = s.negatives(num_neg=6, rng=np.random.default_rng(3))
    assert len(u) == len(train_u) * 6
    train_pairs = set(zip(train_u.tolist(), train_i.tolist()))
    assert not any((a, b) in train_pairs for a, b in zip(u.tolist(), j.tolist()))
    # roughly uniform over the allowed items
    counts = np.bincount(j[u == 0], minlength=n_items)
    allowed = np.setdiff1d(np.arange(n_items), train_i[train_u == 0])
    assert (counts[allowed] > 0).all() and counts[train_i[train_u == 0]].sum() == 0


def test_epoch_has_the_right_mix_of_positives_and_negatives():
    train_u = np.array([0, 0, 1, 1, 1])
    train_i = np.array([0, 1, 2, 3, 4])
    s = D.NegativeSampler(train_u, train_i, n_items=50)
    u, i, y = s.epoch(num_neg=4, rng=np.random.default_rng(0))
    assert len(u) == len(i) == len(y) == 5 * 5
    assert y.sum() == 5 and y.dtype == np.float32
    pos = set(zip(u[y == 1].tolist(), i[y == 1].tolist()))
    assert pos == set(zip(train_u.tolist(), train_i.tolist()))


# ------------------------------------------------------------- real data (if built)
REAL = os.path.exists(D.CACHE.format(seed=0, neg=99))


@pytest.mark.skipif(not REAL, reason="run `python data.py` to build the MovieLens split")
def test_movielens_split_matches_the_paper():
    d = D.load_ml1m()
    assert (d["n_users"], d["n_items"]) == (6040, 3706)                     # paper Table 1
    assert len(d["train_u"]) + len(d["test_u"]) == 1_000_209
    assert np.array_equal(d["test_u"], np.arange(6040))                     # one per user
    assert d["test_candidates"].shape == (6040, 100)
    assert np.array_equal(d["test_candidates"][:, 0], d["test_i"])
    # no test candidate (positive or negative) is a training item for that user
    keys = np.unique(d["train_u"] * d["n_items"] + d["train_i"])
    cand = d["test_candidates"]
    q = (np.arange(6040)[:, None] * d["n_items"] + cand).ravel()
    assert not D.in_sorted(q, keys).any()
    assert all(len(set(row)) == 100 for row in cand.tolist())
