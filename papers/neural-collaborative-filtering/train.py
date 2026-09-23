"""
Train GMF / MLP / NeuMF on MovieLens-1M and evaluate HR@10 and NDCG@10 every epoch.

    python train.py --model gmf   --factors 8
    python train.py --model mlp   --factors 8
    python train.py --model neumf --factors 8                    # from scratch
    python train.py --model neumf --factors 8 --pretrain         # from the GMF + MLP above

Defaults follow the authors' released code: 20 epochs, batch 256, Adam lr 1e-3, 4
negatives per positive, BCE loss. With --pretrain the paper switches to vanilla SGD
(Section 3.4.1), because Adam's moment estimates are not saved with the pre-trained
weights and restarting them would undo the initialisation.

Model selection. The authors' code scores the *test* set after every epoch and reports
the best epoch, which picks the epoch using test labels. This script records every
epoch's test score but reports two numbers: the final epoch (no selection, what a fair
comparison should use) and the best epoch (the paper's protocol, optimistic). Pre-training
uses the final-epoch GMF and MLP, so no test information leaks into NeuMF.

--patience stops once validation HR has not improved for that many epochs, and makes the
saved checkpoint the best-validation one rather than the last. Both need --validate: the
point is to stop without consulting the test set. At 64 factors the models peak by epoch
5-10, so this saves more than half the training time and saves the right weights.

--validate adds the honest third option: hold out each user's second newest interaction
as a validation set, choose the epoch by validation HR, and report the test score at that
epoch. It costs one interaction per user of training data, so its numbers are not
directly comparable to the paper's, but it needs no test labels to pick an epoch. This
matters at 64 factors, where the models peak around epoch 8-10 and then decline.

--reg adds L2 on the embeddings. The authors' Keras code regularises the whole embedding
matrix every step (embeddings_regularizer=l2); this penalises only the rows the batch
actually used, which is the form that works with sparse gradients and weights each row by
how often it is seen. The paper sweeps this and reports 0 for its results, but its models
do not overfit the way these do at 64 factors.

Writes checkpoints/{tag}.pt and results.json[tag].
"""

import argparse
import os
import time

import torch  # before numpy-heavy imports: Anaconda OpenMP load order (utils/plotting.py)
import numpy as np
import torch.nn as nn

import _bootstrap  # noqa: F401
from utils import count_parameters, seed_everything
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.results import update_results
from data import NegativeSampler, load_ml1m
from metrics import evaluate
from model import build, split_parameters

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULTS = os.path.join(HERE, "results.json")


class EarlyStopping:
    """Stop when the monitored score has not improved for `patience` epochs.

    patience=0 disables it. `update` returns True when training should stop; `best_epoch`
    and `best` track the high-water mark, which is also the checkpoint worth keeping.
    """

    def __init__(self, patience=0):
        self.patience = patience
        self.best = float("-inf")
        self.best_epoch = 0
        self.since_best = 0

    def update(self, epoch, score):
        if score > self.best:
            self.best, self.best_epoch, self.since_best = score, epoch, 0
        else:
            self.since_best += 1
        return bool(self.patience) and self.since_best >= self.patience


def run_tag(model, factors, pretrain=False, suffix="", reg=0.0):
    return (f"{model}_f{factors}" + ("_pretrain" if pretrain else "")
            + (f"_reg{reg:g}" if reg else "") + suffix)


def load_model(tag, n_users, n_items):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{tag}.pt"))
    m = build(ckpt["args"]["model"], n_users, n_items, ckpt["args"]["factors"])
    m.load_state_dict(ckpt["model"])
    return m


def train(args):
    assert not args.patience or args.validate, "--patience needs --validate: stopping on the "\
                                               "test set would be selection on test labels"
    seed_everything(args.seed)
    d = load_ml1m(validation=args.validate)
    n_users, n_items = d["n_users"], d["n_items"]
    sampler = NegativeSampler(d["train_u"], d["train_i"], n_items)
    rng = np.random.default_rng(args.seed)

    model = build(args.model, n_users, n_items, args.factors, sparse=args.sparse)
    if args.pretrain:
        assert args.model == "neumf", "--pretrain applies to NeuMF"
        gmf = load_model(run_tag("gmf", args.factors), n_users, n_items)
        mlp = load_model(run_tag("mlp", args.factors), n_users, n_items)
        model.load_pretrained(gmf, mlp, alpha=args.alpha)
    tag = run_tag(args.model, args.factors, args.pretrain, args.tag_suffix, args.reg)

    optimizer = args.optimizer or ("sgd" if args.pretrain else "adam")
    # SGD handles sparse gradients directly. Adam does not: it needs SparseAdam for the
    # embedding tables and plain Adam for the handful of dense layers.
    if optimizer == "sgd":
        opts = [torch.optim.SGD(model.parameters(), lr=args.lr)]
    elif args.sparse:
        emb, dense = split_parameters(model)
        opts = [torch.optim.SparseAdam(emb, lr=args.lr), torch.optim.Adam(dense, lr=args.lr)]
    else:
        opts = [torch.optim.Adam(model.parameters(), lr=args.lr)]
    loss_fn = nn.BCEWithLogitsLoss()   # the paper's log loss (Eq. 7), numerically stable

    def score(m):
        """(test HR, test NDCG, val HR or None)."""
        t_hr, t_nd = evaluate(m, d["test_candidates"])
        v_hr = evaluate(m, d["val_candidates"])[0] if args.validate else None
        return t_hr, t_nd, v_hr

    n_params = count_parameters(model)
    hr, nd, v_hr = score(model)
    print(f"{tag}: {n_params:,} params, {optimizer}{' sparse' if args.sparse else ''} "
          f"lr {args.lr} | "
          f"epoch 0 HR@10 {hr:.4f} NDCG@10 {nd:.4f}", flush=True)
    history = [{"epoch": 0, "hr": hr, "ndcg": nd, "val_hr": v_hr, "loss": None, "seconds": 0.0}]

    stopper = EarlyStopping(args.patience)
    best_state = None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        u, i, y = sampler.epoch(args.num_neg, rng)
        u, i, y = torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(y)
        total, n = 0.0, 0
        for s in range(0, len(u), args.batch_size):
            ub, ib = u[s : s + args.batch_size], i[s : s + args.batch_size]
            logits = model(ub, ib)
            loss = loss_fn(logits, y[s : s + args.batch_size])
            if args.reg:
                loss = loss + args.reg * model.l2_penalty(ub, ib) / len(ub)
            for o in opts:
                o.zero_grad()
            loss.backward()
            for o in opts:
                o.step()
            total += loss.item() * logits.numel()
            n += logits.numel()
        hr, nd, v_hr = score(model)
        rec = {"epoch": epoch, "hr": hr, "ndcg": nd, "val_hr": v_hr,
               "loss": total / n, "seconds": time.time() - t0}
        history.append(rec)
        print(f"  epoch {epoch:2d} | loss {rec['loss']:.4f} | HR@10 {hr:.4f} | NDCG@10 {nd:.4f}"
              + (f" | val HR {v_hr:.4f}" if v_hr is not None else "")
              + f" | {rec['seconds']:4.0f}s", flush=True)
        if args.validate:
            stop = stopper.update(epoch, v_hr)
            if stopper.best_epoch == epoch:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if stop:
                print(f"  early stop: val HR has not improved since epoch "
                      f"{stopper.best_epoch}", flush=True)
                break

    best = max(history[1:], key=lambda r: r["hr"])
    final = history[-1]
    chosen = max(history[1:], key=lambda r: r["val_hr"]) if args.validate else None
    # With a validation set the checkpoint worth keeping is the best one, not the last.
    if best_state is not None:
        model.load_state_dict(best_state)
    minutes = (time.time() - t_start) / 60
    print(f"{tag} done in {minutes:.1f} min | final HR {final['hr']:.4f} NDCG {final['ndcg']:.4f} "
          f"| best-on-test (epoch {best['epoch']}) HR {best['hr']:.4f} NDCG {best['ndcg']:.4f}"
          + (f" | val-selected (epoch {chosen['epoch']}) HR {chosen['hr']:.4f} "
             f"NDCG {chosen['ndcg']:.4f}" if chosen else ""))

    os.makedirs(CKPT_DIR, exist_ok=True)
    save_checkpoint({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs},
                    os.path.join(CKPT_DIR, f"{tag}.pt"))
    update_results(RESULTS, {tag: {
        "config": {**vars(args), "optimizer": optimizer}, "parameters": n_params,
        "final": {"epoch": final["epoch"], "hr": final["hr"], "ndcg": final["ndcg"]},
        "best_on_test": {"epoch": best["epoch"], "hr": best["hr"], "ndcg": best["ndcg"]},
        **({"val_selected": {"epoch": chosen["epoch"], "hr": chosen["hr"],
                             "ndcg": chosen["ndcg"], "val_hr": chosen["val_hr"]}} if chosen else {}),
        "epochs_run": len(history) - 1, "saved_epoch": stopper.best_epoch if best_state else final["epoch"],
        "minutes": minutes, "history": history}})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gmf", "mlp", "neumf"], required=True)
    p.add_argument("--factors", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-neg", type=int, default=4)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default=None,
                   help="default: adam, or sgd with --pretrain (as in the paper)")
    p.add_argument("--pretrain", action="store_true")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--reg", type=float, default=0.0,
                   help="L2 on the embedding rows used by each batch (0 = the paper's setting)")
    p.add_argument("--tag-suffix", default="", help="distinguish otherwise identical runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=0,
                   help="stop after N epochs without a validation improvement (needs --validate)")
    p.add_argument("--validate", action="store_true",
                   help="hold out a validation interaction per user and pick the epoch with it")
    p.add_argument("--sparse", action="store_true",
                   help="sparse embedding gradients + SparseAdam; faster above ~32 factors")
    train(p.parse_args())
