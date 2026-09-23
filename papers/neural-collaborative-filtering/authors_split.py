"""
Compare this repo's MovieLens split with the one the authors published, and score a
trained model under theirs.

The 64-factor results here fall short of the paper while the 8-factor ones match. One
possible cause is the data itself: this pipeline rebuilds the leave-one-out split from
the raw ratings, whereas the paper's numbers come from the fixed files in the authors'
repository (Data/ml-1m.{train.rating,test.rating,test.negative}). If those disagree with
ours, the comparison was never like-for-like.

    python authors_split.py                       # download and compare the two splits
    python authors_split.py --evaluate gmf_f64    # also score a checkpoint under theirs

Their files use their own contiguous ids. The comparison first checks whether their id
mapping agrees with ours (each user's set of interacted items must match); only then are
held-out items comparable.
"""

import argparse
import os

import torch  # before numpy-heavy imports: Anaconda OpenMP load order
import numpy as np
import requests

import _bootstrap  # noqa: F401
from utils.results import update_results
from data import DATA_DIR, load_ml1m
from metrics import evaluate
from train import RESULTS, load_model

BASE = "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data"
FILES = ("train.rating", "test.rating", "test.negative")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {}
    for name in FILES:
        p = os.path.join(DATA_DIR, f"authors_ml-1m.{name}")
        if not os.path.exists(p):
            url = f"{BASE}/ml-1m.{name}"
            print(f"downloading {url}")
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with open(p, "wb") as f:
                f.write(r.content)
        paths[name] = p
    return paths


def load_authors():
    """Returns train_u, train_i, test_u, test_i, negatives (n_users, 99)."""
    p = download()
    tr = np.loadtxt(p["train.rating"], dtype=np.int64, usecols=(0, 1))
    te = np.loadtxt(p["test.rating"], dtype=np.int64, usecols=(0, 1))
    negs = []
    with open(p["test.negative"]) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            negs.append([int(x) for x in parts[1:]])
    return tr[:, 0], tr[:, 1], te[:, 0], te[:, 1], np.array(negs, dtype=np.int64)


def item_sets(users, items, n_users):
    order = np.argsort(users, kind="stable")
    u, i = users[order], items[order]
    starts = np.searchsorted(u, np.arange(n_users), side="left")
    ends = np.searchsorted(u, np.arange(n_users), side="right")
    return [frozenset(i[s:e].tolist()) for s, e in zip(starts, ends)]


def match_items(ours_u, ours_i, their_u, their_i, n_items):
    """Recover the correspondence between their item ids and ours.

    Their files carry their own contiguous ids, so a direct comparison is meaningless.
    But if the underlying interactions are the same data, each item is identified by the
    set of users who rated it, and that signature is shared between the two id spaces.
    Returns (their_id -> our_id, number of items whose signature is shared), with the
    mapping None if no bijection exists, which would mean these are not the same
    interactions.
    """
    def by_item(u, i):
        order = np.argsort(i, kind="stable")
        u_s, i_s = u[order], i[order]
        starts = np.searchsorted(i_s, np.arange(n_items), side="left")
        ends = np.searchsorted(i_s, np.arange(n_items), side="right")
        return [frozenset(u_s[a:b].tolist()) for a, b in zip(starts, ends)]

    from collections import defaultdict
    index = defaultdict(list)
    for item, sig in enumerate(by_item(ours_u, ours_i)):
        index[sig].append(item)
    # A handful of MovieLens items are rated by exactly the same set of users (36 of
    # 3,706). Those are indistinguishable from the interactions alone, so any consistent
    # assignment within such a group is a valid bijection; the count is returned so the
    # caller can say how many test-item comparisons are ambiguous.
    ambiguous = sum(len(v) for v in index.values() if len(v) > 1)
    mapping = np.full(n_items, -1, dtype=np.int64)
    for their_item, sig in enumerate(by_item(their_u, their_i)):
        if not index.get(sig):
            return None, ambiguous
        mapping[their_item] = index[sig].pop()
    return (mapping if len(set(mapping.tolist())) == n_items else None), ambiguous


def compare(ours, theirs):
    a_u, a_i, a_te_u, a_te_i, a_neg = theirs
    n_users, n_items = ours["n_users"], ours["n_items"]
    report = {
        "authors_train_interactions": int(len(a_u)),
        "our_train_interactions": int(len(ours["train_u"])),
        "authors_users": int(a_u.max() + 1),
        "authors_max_item_id": int(max(a_i.max(), a_neg.max())),
        "authors_negatives_per_user": int(a_neg.shape[1]),
    }

    # Do the two id mappings agree? Compare each user's full interacted set.
    ours_all_u = np.concatenate([ours["train_u"], ours["test_u"]])
    ours_all_i = np.concatenate([ours["train_i"], ours["test_i"]])
    theirs_all_u = np.concatenate([a_u, a_te_u])
    theirs_all_i = np.concatenate([a_i, a_te_i])
    mine = item_sets(ours_all_u, ours_all_i, n_users)
    yours = item_sets(theirs_all_u, theirs_all_i, n_users)
    same_mapping = sum(m == y for m, y in zip(mine, yours))
    report["users_with_identical_item_sets"] = int(same_mapping)
    report["id_mappings_agree"] = bool(same_mapping == n_users)

    # Their ids are their own, so recover the correspondence from each item's user set.
    if report["id_mappings_agree"]:
        mapping, ambiguous = np.arange(n_items), 0
    else:
        mapping, ambiguous = match_items(ours_all_u, ours_all_i, theirs_all_u,
                                         theirs_all_i, n_items)
    report["same_underlying_interactions"] = mapping is not None
    report["items_with_a_shared_user_set"] = int(ambiguous)
    if mapping is None:
        return report

    order = np.argsort(a_te_u, kind="stable")
    their_test = mapping[a_te_i[order]]
    report["test_items_identical"] = int((their_test == ours["test_i"]).sum())
    report["test_items_identical_fraction"] = float((their_test == ours["test_i"]).mean())
    overlap = [len(set(mapping[a_neg[u]].tolist()) & set(ours["test_candidates"][u, 1:].tolist()))
               for u in range(n_users)]
    report["mean_negative_overlap_of_99"] = float(np.mean(overlap))
    return report, mapping


def authors_candidates(theirs, mapping):
    """(n_users, 100) with the authors' held-out item in column 0, in user order and
    translated into our item ids."""
    _, _, te_u, te_i, neg = theirs
    order = np.argsort(te_u, kind="stable")
    return mapping[np.concatenate([te_i[order][:, None], neg[order]], axis=1)]


def load_as_split():
    """The authors' files in the same shape as data.load_ml1m, in their own id space.

    Training a model on this is the only valid way to compare with their numbers:
    evaluating a model trained on our split against their held-out items leaks, because
    the two splits disagree for 27.6% of users and every one of those items is in our
    training set.
    """
    tr_u, tr_i, te_u, te_i, neg = load_authors()
    order = np.argsort(te_u, kind="stable")
    n_users = int(max(tr_u.max(), te_u.max())) + 1
    n_items = int(max(tr_i.max(), te_i.max(), neg.max())) + 1
    return {
        "n_users": n_users, "n_items": n_items,
        "train_u": tr_u, "train_i": tr_i, "test_u": te_u[order], "test_i": te_i[order],
        "test_candidates": np.concatenate([te_i[order][:, None], neg[order]], axis=1),
    }


def main(args):
    ours = load_ml1m()
    theirs = load_authors()
    out = compare(ours, theirs)
    report, mapping = out if isinstance(out, tuple) else (out, None)
    width = max(len(k) for k in report)
    for k, v in report.items():
        print(f"{k:<{width}}  {v}")

    if args.evaluate:
        assert mapping is not None, "cannot score under their split without an id mapping"
        model = load_model(args.evaluate, ours["n_users"], ours["n_items"])
        ours_hr, ours_nd = evaluate(model, ours["test_candidates"])
        their_hr, their_nd = evaluate(model, authors_candidates(theirs, mapping))
        print(f"\n{args.evaluate} scored on our split      : HR@10 {ours_hr:.4f}  NDCG@10 {ours_nd:.4f}")
        print(f"{args.evaluate} scored on authors' split : HR@10 {their_hr:.4f}  NDCG@10 {their_nd:.4f}")
        report[f"{args.evaluate}_ours"] = {"hr": ours_hr, "ndcg": ours_nd}
        report[f"{args.evaluate}_authors"] = {"hr": their_hr, "ndcg": their_nd}

    update_results(RESULTS, report, namespace="authors_split")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--evaluate", help="a run tag in checkpoints/, e.g. gmf_f64")
    main(p.parse_args())
