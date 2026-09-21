"""
MovieLens-1M in the paper's implicit-feedback, leave-one-out setup (Section 4.1).

    6,040 users, 3,706 items, 1,000,209 ratings. Every rating becomes a positive
    interaction; the star value is discarded. For each user the latest interaction
    (by timestamp) is held out for testing and the rest are training data. At test
    time the held-out item is ranked against 99 items the user never interacted with.

The split and the test negatives are computed once with a fixed seed and cached, so
every model is evaluated on exactly the same 604,000 (user, item) pairs.

Training negatives are resampled every epoch (4 per positive in the paper) and are
rejected only if they appear in the user's *training* set, as in the authors' code.
Rejecting the held-out test item too would use test labels during training.

    from data import load_ml1m, NegativeSampler
    d = load_ml1m()
    sampler = NegativeSampler(d["train_u"], d["train_i"], d["n_items"])
    u, i, y = sampler.epoch(num_neg=4, rng=np.random.default_rng(0))
"""

import io
import os
import zipfile

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
CACHE = os.path.join(DATA_DIR, "ml1m_loo_seed{seed}_neg{neg}.npz")


# ---------------------------------------------------------------------------- raw data
def download_ml1m():
    """Return raw (user, item, timestamp) int arrays in file order, ids as in the file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = os.path.join(DATA_DIR, "ratings.dat")
    if not os.path.exists(raw):
        print(f"downloading {URL}")
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z, open(raw, "wb") as out:
            out.write(z.read("ml-1m/ratings.dat"))
    arr = _read_ratings(raw)          # columns: UserID, MovieID, Rating, Timestamp
    return arr[:, 0], arr[:, 1], arr[:, 3]


def _read_ratings(path):
    # np.loadtxt cannot split on a multi-character delimiter; this is still ~1 s.
    with open(path, encoding="latin-1") as f:
        rows = [line.split("::") for line in f]
    return np.array(rows, dtype=np.int64)


# ------------------------------------------------------------------------ pure logic
def reindex(ids):
    """Map arbitrary ids to 0..n-1 in sorted order. Returns (new_ids, n)."""
    uniq, inv = np.unique(ids, return_inverse=True)
    return inv.astype(np.int64), len(uniq)


def leave_one_out(users, items, timestamps):
    """Hold out each user's latest interaction.

    Ties in timestamp are broken by position in the input, so the split is fully
    deterministic. Returns train_u, train_i, test_u, test_i with test sorted by user.
    """
    order = np.lexsort((np.arange(len(users)), timestamps, users))   # user, then time, then file order
    u, i = users[order], items[order]
    last = np.r_[u[1:] != u[:-1], True]                             # last row of each user's block
    return u[~last], i[~last], u[last], i[last]


def sample_test_negatives(users, items, test_u, n_items, num_neg=99, seed=0):
    """For each test user, `num_neg` distinct items the user never interacted with
    anywhere (train or test). Returns an (n_test_users, num_neg) array aligned with test_u."""
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    u_sorted, i_sorted = users[order], items[order]
    starts = np.searchsorted(u_sorted, test_u, side="left")
    ends = np.searchsorted(u_sorted, test_u, side="right")
    all_items = np.arange(n_items)
    out = np.empty((len(test_u), num_neg), dtype=np.int64)
    for row, (s, e) in enumerate(zip(starts, ends)):
        seen = i_sorted[s:e]
        pool = np.setdiff1d(all_items, seen, assume_unique=False)
        out[row] = rng.choice(pool, size=num_neg, replace=False)
    return out


def in_sorted(queries, keys):
    """Vectorised membership test: is each query present in the sorted array `keys`?"""
    idx = np.searchsorted(keys, queries)
    idx = np.minimum(idx, len(keys) - 1)
    return keys[idx] == queries


class NegativeSampler:
    """Draws training negatives uniformly from items not in the user's training set.

    Rejection sampling, vectorised: draw all candidates at once, find the ones that
    collide with a training (user, item) pair through a binary search on encoded keys,
    redraw only those, repeat. At MovieLens density (~4.4% of items per user) almost
    all draws succeed first time, so 4 million negatives take well under a second.
    The authors' code does this with a per-sample Python loop and a dict lookup.
    """

    def __init__(self, train_u, train_i, n_items):
        self.train_u = np.asarray(train_u, dtype=np.int64)
        self.train_i = np.asarray(train_i, dtype=np.int64)
        self.n_items = n_items
        self.keys = np.unique(self.train_u * n_items + self.train_i)

    def negatives(self, num_neg, rng):
        """Returns (u, j): num_neg negatives for every training positive."""
        u = np.repeat(self.train_u, num_neg)
        j = rng.integers(0, self.n_items, size=u.size)
        bad = in_sorted(u * self.n_items + j, self.keys)
        while bad.any():
            j[bad] = rng.integers(0, self.n_items, size=int(bad.sum()))
            still = in_sorted(u[bad] * self.n_items + j[bad], self.keys)
            bad[bad] = still
        return u, j

    def epoch(self, num_neg, rng):
        """One epoch of training data: all positives plus fresh negatives, shuffled.
        Returns (users, items, labels) as numpy arrays (labels float32 0/1)."""
        nu, nj = self.negatives(num_neg, rng)
        u = np.concatenate([self.train_u, nu])
        i = np.concatenate([self.train_i, nj])
        y = np.concatenate([np.ones(len(self.train_u), np.float32), np.zeros(len(nu), np.float32)])
        perm = rng.permutation(len(u))
        return u[perm], i[perm], y[perm]


# ------------------------------------------------------------------------------ entry
def load_ml1m(seed=0, num_test_neg=99):
    """Returns a dict with the leave-one-out split and fixed test candidates:

        n_users, n_items, train_u, train_i, test_u, test_i,
        test_candidates  (n_users, 1 + num_test_neg): column 0 is the held-out item.
    """
    path = CACHE.format(seed=seed, neg=num_test_neg)
    if os.path.exists(path):
        z = np.load(path)
        return {k: (int(z[k]) if z[k].ndim == 0 else z[k]) for k in z.files}

    users, items, ts = download_ml1m()
    users, n_users = reindex(users)
    items, n_items = reindex(items)
    train_u, train_i, test_u, test_i = leave_one_out(users, items, ts)
    neg = sample_test_negatives(users, items, test_u, n_items, num_test_neg, seed)
    d = {
        "n_users": n_users, "n_items": n_items,
        "train_u": train_u, "train_i": train_i, "test_u": test_u, "test_i": test_i,
        "test_candidates": np.concatenate([test_i[:, None], neg], axis=1),
    }
    np.savez_compressed(path, **d)
    return d


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = load_ml1m()
    n = len(d["train_u"]) + len(d["test_u"])
    print(f"users {d['n_users']}  items {d['n_items']}  interactions {n:,}  "
          f"sparsity {1 - n / (d['n_users'] * d['n_items']):.2%}  ({time.time() - t0:.1f}s)")
    print(f"train {len(d['train_u']):,}  test {len(d['test_u']):,}  "
          f"test candidates {d['test_candidates'].shape}")
    s = NegativeSampler(d["train_u"], d["train_i"], d["n_items"])
    t0 = time.time()
    u, i, y = s.epoch(4, np.random.default_rng(0))
    print(f"one epoch: {len(u):,} examples, {y.mean():.0%} positive, sampled in {time.time() - t0:.2f}s")
