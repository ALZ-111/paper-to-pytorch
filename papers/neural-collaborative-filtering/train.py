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
from model import build

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULTS = os.path.join(HERE, "results.json")


def run_tag(model, factors, pretrain=False):
    return f"{model}_f{factors}" + ("_pretrain" if pretrain else "")


def load_model(tag, n_users, n_items):
    ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"{tag}.pt"))
    m = build(ckpt["args"]["model"], n_users, n_items, ckpt["args"]["factors"])
    m.load_state_dict(ckpt["model"])
    return m


def train(args):
    seed_everything(args.seed)
    d = load_ml1m()
    n_users, n_items = d["n_users"], d["n_items"]
    sampler = NegativeSampler(d["train_u"], d["train_i"], n_items)
    rng = np.random.default_rng(args.seed)

    model = build(args.model, n_users, n_items, args.factors)
    if args.pretrain:
        assert args.model == "neumf", "--pretrain applies to NeuMF"
        gmf = load_model(run_tag("gmf", args.factors), n_users, n_items)
        mlp = load_model(run_tag("mlp", args.factors), n_users, n_items)
        model.load_pretrained(gmf, mlp, alpha=args.alpha)
    tag = run_tag(args.model, args.factors, args.pretrain)

    optimizer = args.optimizer or ("sgd" if args.pretrain else "adam")
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()   # the paper's log loss (Eq. 7), numerically stable

    n_params = count_parameters(model)
    hr, nd = evaluate(model, d["test_candidates"])
    print(f"{tag}: {n_params:,} params, {optimizer} lr {args.lr} | "
          f"epoch 0 HR@10 {hr:.4f} NDCG@10 {nd:.4f}", flush=True)
    history = [{"epoch": 0, "hr": hr, "ndcg": nd, "loss": None, "seconds": 0.0}]

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        u, i, y = sampler.epoch(args.num_neg, rng)
        u, i, y = torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(y)
        total, n = 0.0, 0
        for s in range(0, len(u), args.batch_size):
            logits = model(u[s : s + args.batch_size], i[s : s + args.batch_size])
            loss = loss_fn(logits, y[s : s + args.batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * logits.numel()
            n += logits.numel()
        hr, nd = evaluate(model, d["test_candidates"])
        rec = {"epoch": epoch, "hr": hr, "ndcg": nd, "loss": total / n, "seconds": time.time() - t0}
        history.append(rec)
        print(f"  epoch {epoch:2d} | loss {rec['loss']:.4f} | HR@10 {hr:.4f} | NDCG@10 {nd:.4f} "
              f"| {rec['seconds']:4.0f}s", flush=True)

    best = max(history[1:], key=lambda r: r["hr"])
    final = history[-1]
    minutes = (time.time() - t_start) / 60
    print(f"{tag} done in {minutes:.1f} min | final HR {final['hr']:.4f} NDCG {final['ndcg']:.4f} "
          f"| best (epoch {best['epoch']}) HR {best['hr']:.4f} NDCG {best['ndcg']:.4f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    save_checkpoint({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs},
                    os.path.join(CKPT_DIR, f"{tag}.pt"))
    update_results(RESULTS, {tag: {
        "config": {**vars(args), "optimizer": optimizer}, "parameters": n_params,
        "final": {"epoch": final["epoch"], "hr": final["hr"], "ndcg": final["ndcg"]},
        "best_on_test": {"epoch": best["epoch"], "hr": best["hr"], "ndcg": best["ndcg"]},
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
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())
