"""
Throughput benchmark for the Transformer on this machine.

    python bench.py                       # training + inference at the default thread count
    python bench.py --threads 1 2 4 8     # sweep torch.set_num_threads
    python bench.py --json bench.json     # also write results

Training: real Multi30k batches (word-level, max_tokens 2500), forward + backward + Adam
step, reported as target tokens/s after a warm-up. Inference: greedy decode of the 1,000
test sentences at batch 128 with the KV cache, reported as sentences/s. Same model config
as translate.py's defaults (3 layers, d_model 256), random weights, so no checkpoint needed.
"""

import argparse
import json
import time

import torch
import torch.nn as nn

from data import BOS, EOS, PAD, batches_by_length, collate, load_multi30k
from model import Transformer


def bench_train(model, pairs, max_tokens, steps, warmup=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    model.train()
    it = batches_by_length(pairs, max_tokens)
    tokens, t0 = 0, None
    for i in range(steps + warmup):
        try:
            src, tgt = next(it)
        except StopIteration:
            it = batches_by_length(pairs, max_tokens)
            src, tgt = next(it)
        if i == warmup:
            t0 = time.perf_counter()
        logits = model(src, tgt[:, :-1])
        loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i >= warmup:
            tokens += int((tgt[:, 1:] != PAD).sum())
    return tokens / (time.perf_counter() - t0)


@torch.no_grad()
def bench_infer(model, pairs, batch_size=128, max_len=60):
    model.eval()
    # warm-up on one batch
    src, _ = collate(pairs[:batch_size])
    model.greedy_decode(src, BOS, EOS, max_len=max_len)
    t0 = time.perf_counter()
    for i in range(0, len(pairs), batch_size):
        src, _ = collate(pairs[i : i + batch_size])
        model.greedy_decode(src, BOS, EOS, max_len=max_len)
    return len(pairs) / (time.perf_counter() - t0)


def run(threads, train_pairs, test_pairs, args):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = Transformer(7882, 5898, d_model=256, n_layers=3, n_heads=8, d_ff=1024,
                        dropout=0.1, pad_idx=PAD, tie_weights=True)
    tr = bench_train(model, train_pairs, args.max_tokens, args.steps)
    inf = bench_infer(model, test_pairs)
    return {"threads": threads, "train_tokens_per_s": tr, "infer_sentences_per_s": inf}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, nargs="+", default=[torch.get_num_threads()])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--json")
    args = p.parse_args()

    train_pairs, _, test_pairs, _, _ = load_multi30k()
    print(f"{'threads':>7} {'train tok/s':>12} {'infer sent/s':>13}")
    results = []
    for t in args.threads:
        r = run(t, train_pairs, test_pairs, args)
        results.append(r)
        print(f"{t:>7} {r['train_tokens_per_s']:>12,.0f} {r['infer_sentences_per_s']:>13,.1f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
