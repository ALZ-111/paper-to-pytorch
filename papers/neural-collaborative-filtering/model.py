"""
Neural Collaborative Filtering - He et al., 2017
Paper: https://arxiv.org/abs/1708.05031

The three models of Section 3, returning *logits* (sigmoid is folded into the loss,
BCEWithLogitsLoss, which is the paper's log loss computed stably):

    GMF    y = h^T (p_u * q_i)                         Eq. 9, generalised MF
    MLP    y = h^T tower([p_u ; q_i])                  Eq. 10, ReLU tower, halving widths
    NeuMF  y = h^T [p_u^G * q_i^G ; tower([p_u^M ; q_i^M])]    Eq. 11-12

NeuMF gives each branch its own embeddings: the paper found sharing them limits the
model, because the best embedding size differs between the two branches.

Sizes follow the authors' released code: with 8 predictive factors, GMF embeddings are 8
wide and the MLP tower is 64 -> 32 -> 16 -> 8 on 32-wide user and item embeddings. (The
paper's prose gives a 32 -> 16 -> 8 example; the code and the paper's "three hidden
layers" both say 64 -> 32 -> 16 -> 8.)
"""

import math

import torch
import torch.nn as nn


def _init_embedding(emb):
    nn.init.normal_(emb.weight, std=0.01)            # authors' code: normal(scale=0.01)


def embedding(n, dim, sparse=False):
    """An embedding table whose gradient is dense (default) or sparse.

    A sparse gradient carries only the rows a batch touched, which lets SparseAdam skip
    the rest of the table. That is a large saving when the tables dominate the parameter
    count (they are 99% of the MLP's), but it is not the same update: see train.py.
    """
    emb = nn.Embedding(n, dim, sparse=sparse)
    _init_embedding(emb)
    return emb


def _init_output(linear):
    # Keras lecun_uniform, as in the authors' code: U(-sqrt(3/fan_in), +sqrt(3/fan_in)).
    bound = math.sqrt(3.0 / linear.in_features)
    nn.init.uniform_(linear.weight, -bound, bound)
    nn.init.zeros_(linear.bias)


class GMF(nn.Module):
    """Generalised matrix factorisation. With h fixed to all-ones and no bias this is
    exactly MF's dot product; learning h lets factors be weighted unequally."""

    def __init__(self, n_users, n_items, factors=8, sparse=False):
        super().__init__()
        self.user = embedding(n_users, factors, sparse)
        self.item = embedding(n_items, factors, sparse)
        self.out = nn.Linear(factors, 1)
        _init_output(self.out)

    def features(self, u, i):
        """(B,) user ids, (B,) item ids -> (B, factors) interaction vector."""
        return self.user(u) * self.item(i)

    def l2_penalty(self, u, i):
        """Sum of squares of the embedding rows this batch used (see train.py --reg)."""
        return self.user(u).pow(2).sum() + self.item(i).pow(2).sum()

    def forward(self, u, i):
        return self.out(self.features(u, i)).squeeze(-1)


class MLP(nn.Module):
    """Concatenate user and item embeddings and learn their interaction with a ReLU
    tower. layers[0] is the concatenated width, so each embedding is layers[0] // 2."""

    def __init__(self, n_users, n_items, layers=(64, 32, 16, 8), sparse=False):
        super().__init__()
        assert layers[0] % 2 == 0, "layers[0] is the concatenated embedding width"
        self.user = embedding(n_users, layers[0] // 2, sparse)
        self.item = embedding(n_items, layers[0] // 2, sparse)
        tower = []
        for a, b in zip(layers[:-1], layers[1:]):
            lin = nn.Linear(a, b)
            nn.init.xavier_uniform_(lin.weight)      # Keras Dense default (glorot_uniform)
            nn.init.zeros_(lin.bias)
            tower += [lin, nn.ReLU()]
        self.tower = nn.Sequential(*tower)
        self.out = nn.Linear(layers[-1], 1)
        _init_output(self.out)

    def features(self, u, i):
        return self.tower(torch.cat([self.user(u), self.item(i)], dim=-1))

    def l2_penalty(self, u, i):
        return self.user(u).pow(2).sum() + self.item(i).pow(2).sum()

    def forward(self, u, i):
        return self.out(self.features(u, i)).squeeze(-1)


class NeuMF(nn.Module):
    """Fusion of GMF and MLP (Section 3.4): each branch computes its interaction vector
    with its own embeddings, the two are concatenated, one output layer scores them."""

    def __init__(self, n_users, n_items, factors=8, layers=(64, 32, 16, 8), sparse=False):
        super().__init__()
        self.gmf = GMF(n_users, n_items, factors, sparse)
        self.mlp = MLP(n_users, n_items, layers, sparse)
        # The branches' own output layers are unused inside NeuMF; remove them so they
        # neither count as parameters nor appear in checkpoints.
        self.gmf.out = None
        self.mlp.out = None
        self.out = nn.Linear(factors + layers[-1], 1)
        _init_output(self.out)

    def forward(self, u, i):
        h = torch.cat([self.gmf.features(u, i), self.mlp.features(u, i)], dim=-1)
        return self.out(h).squeeze(-1)

    def l2_penalty(self, u, i):
        return self.gmf.l2_penalty(u, i) + self.mlp.l2_penalty(u, i)

    @torch.no_grad()
    def load_pretrained(self, gmf, mlp, alpha=0.5):
        """Pre-training (Section 3.4.1): initialise each branch from a trained GMF and MLP,
        and the output layer as h = [alpha * h_GMF ; (1 - alpha) * h_MLP].

        The bias is blended the same way, which makes the freshly initialised NeuMF's
        logit exactly alpha * GMF_logit + (1 - alpha) * MLP_logit: training starts from an
        ensemble of the two, not from noise. The paper uses alpha = 0.5.
        """
        assert gmf.out is not None and mlp.out is not None, "pass trained GMF/MLP models"
        self.gmf.load_state_dict({k: v for k, v in gmf.state_dict().items()
                                  if not k.startswith("out.")})
        self.mlp.load_state_dict({k: v for k, v in mlp.state_dict().items()
                                  if not k.startswith("out.")})
        self.out.weight.copy_(torch.cat([alpha * gmf.out.weight, (1 - alpha) * mlp.out.weight], dim=1))
        self.out.bias.copy_(alpha * gmf.out.bias + (1 - alpha) * mlp.out.bias)
        return self


def build(kind, n_users, n_items, factors=8, sparse=False):
    """GMF / MLP / NeuMF at a given number of predictive factors, sized as in the paper:
    the MLP tower ends at `factors` and doubles back up for three hidden layers."""
    layers = (8 * factors, 4 * factors, 2 * factors, factors)
    if kind == "gmf":
        return GMF(n_users, n_items, factors, sparse)
    if kind == "mlp":
        return MLP(n_users, n_items, layers, sparse)
    if kind == "neumf":
        return NeuMF(n_users, n_items, factors, layers, sparse)
    raise ValueError(f"unknown model {kind!r}")


def split_parameters(model):
    """(sparse embedding weights, everything else) for building the two optimizers."""
    emb = [p for m in model.modules() if isinstance(m, nn.Embedding) and m.sparse
           for p in m.parameters()]
    ids = {id(p) for p in emb}
    return emb, [p for p in model.parameters() if id(p) not in ids]


if __name__ == "__main__":
    for kind in ("gmf", "mlp", "neumf"):
        m = build(kind, 6040, 3706, factors=8)
        print(f"{kind:6s} {sum(p.numel() for p in m.parameters()):>9,} parameters")
