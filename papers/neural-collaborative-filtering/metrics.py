"""
Top-K ranking metrics for the leave-one-out protocol (Section 4.1).

Each user has one positive (the held-out item) ranked against 99 negatives. With
`rank` the number of negatives scored at or above the positive (0 = ranked first):

    HR@K   = 1 if rank < K else 0                  did the item make the top K?
    NDCG@K = 1 / log2(rank + 2) if rank < K else 0 how high, with a log discount

Ties count *against* the positive. Real models essentially never tie, so this matches
the usual definition, but it stops a degenerate model that scores everything equally
from being awarded a perfect hit ratio, which strict ">" would do.
"""

import torch


def positive_rank(scores):
    """scores: (U, C) with the positive in column 0 -> (U,) number of negatives at or
    above it."""
    return (scores[:, 1:] >= scores[:, :1]).sum(dim=1)


def hit_ratio(rank, k=10):
    return (rank < k).float().mean().item()


def ndcg(rank, k=10):
    gain = 1.0 / torch.log2(rank.float() + 2.0)
    return torch.where(rank < k, gain, torch.zeros_like(gain)).mean().item()


@torch.inference_mode()
def evaluate(model, candidates, k=10, batch_users=2048):
    """Score every (user, candidate) pair and return (HR@k, NDCG@k).

    candidates: (U, C) item ids with the held-out positive in column 0; row u belongs
    to user u. All 604k MovieLens pairs go through the model in a few large batches.
    """
    model.eval()
    cand = torch.as_tensor(candidates, dtype=torch.long)
    U, C = cand.shape
    ranks = []
    for s in range(0, U, batch_users):
        items = cand[s : s + batch_users]
        users = torch.arange(s, s + items.size(0)).unsqueeze(1).expand_as(items)
        scores = model(users.reshape(-1), items.reshape(-1)).view(items.shape)
        ranks.append(positive_rank(scores))
    rank = torch.cat(ranks)
    return hit_ratio(rank, k), ndcg(rank, k)
