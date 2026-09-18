"""
Train and evaluate the Transformer on Multi30k German -> English.

    python translate.py train    [--epochs 15] [--d-model 256] [--accum-steps 4] ...
    python translate.py evaluate [--beam 4]            # test-set BLEU from best checkpoint
    python translate.py demo "ein mann fährt fahrrad ."  # translate a sentence

Checkpoints go to checkpoints/, metrics to results.json (both git-ignored except
results.json, which is small and worth versioning).
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

import _bootstrap  # noqa: F401  (repo root on sys.path)
from utils import count_parameters, get_device, seed_everything
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.results import update_results
from bleu import corpus_bleu
from data import (BOS, EOS, PAD, DATA_DIR, batches_by_length, collate, load_multi30k,
                  to_words, tokenize)
from bpe import BPE
from model import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULTS = os.path.join(HERE, "results.json")


def ckpt_path(tokenizer, name="best", variant=""):
    """Word-level run keeps its original file names; BPE run is namespaced; a
    variant tag (e.g. 'prenorm') is appended after the tokenizer."""
    prefix = "multi30k" if tokenizer == "word" else f"multi30k_{tokenizer}"
    if variant:
        prefix += f"_{variant}"
    return os.path.join(CKPT_DIR, f"{prefix}_{name}.pt")


BEST_CKPT = ckpt_path("word")


class NoamLR:
    """lr = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)  (Section 5.3)"""

    def __init__(self, optimizer, d_model, warmup, factor=1.0):
        self.opt, self.d_model, self.warmup, self.factor = optimizer, d_model, warmup, factor
        self.step_num = 0

    def rate(self):
        s = self.step_num
        return self.factor * self.d_model ** -0.5 * min(s ** -0.5, s * self.warmup ** -1.5)

    def step(self):
        self.step_num += 1
        for g in self.opt.param_groups:
            g["lr"] = self.rate()
        self.opt.step()


def build_model(args, src_vocab_size, tgt_vocab_size):
    return Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        d_ff=args.d_ff, dropout=args.dropout, pad_idx=PAD, tie_weights=True,
        pre_norm=getattr(args, "pre_norm", False),
    )


@torch.no_grad()
def translate_corpus(model, pairs, tgt_vocab, device, beam=0, batch_size=128, max_len=60):
    """Decode every source sentence in `pairs`. Returns list of token lists."""
    model.eval()
    # Batch sentences of similar length together: less padding per batch, and the
    # early-exit in the decoders fires for a whole batch at nearly the same step.
    # Each sentence's result is independent of its batch-mates (the source mask
    # handles padding), so this only changes speed, not output.
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
    hyps = [None] * len(pairs)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        src, _ = collate([pairs[j] for j in idx])
        src = src.to(device)
        if beam > 0:
            outs = model.beam_search(src, BOS, EOS, beam_size=beam, max_len=max_len)
        else:
            outs = model.greedy_decode(src, BOS, EOS, max_len=max_len).tolist()
        for j, row in zip(idx, outs):
            hyps[j] = tgt_vocab.decode(row)
    return hyps


def bleu_on(model, pairs, tgt_vocab, device, beam=0):
    """BLEU on words: BPE pieces are joined back into words before scoring so the
    number is comparable across tokenizers."""
    hyps = [to_words(h) for h in translate_corpus(model, pairs, tgt_vocab, device, beam=beam)]
    refs = [to_words(tgt_vocab.decode(t)) for _, t in pairs]
    return corpus_bleu(hyps, refs), hyps


@torch.no_grad()
def eval_loss(model, pairs, criterion, device, max_tokens):
    model.eval()
    total, n_tok = 0.0, 0
    for src, tgt in batches_by_length(pairs, max_tokens, shuffle=False):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        tokens = (tgt[:, 1:] != PAD).sum().item()
        total += loss.item() * tokens
        n_tok += tokens
    return total / n_tok


def train(args):
    device = get_device()
    seed_everything(args.seed)
    print(f"device: {device}  threads: {torch.get_num_threads()}")

    train_pairs, val_pairs, test_pairs, src_vocab, tgt_vocab = load_multi30k(
        tokenizer=args.tokenizer, bpe_merges=args.bpe_merges)
    print(f"tokenizer {args.tokenizer}  train {len(train_pairs)}  val {len(val_pairs)}  "
          f"test {len(test_pairs)}  src vocab {len(src_vocab)}  tgt vocab {len(tgt_vocab)}")

    model = build_model(args, len(src_vocab), len(tgt_vocab)).to(device)
    n_params = count_parameters(model)
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, args.d_model, warmup=args.warmup, factor=args.lr_factor)
    # Label smoothing eps=0.1 (Section 5.4). Padding positions are excluded.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    # Unsmoothed loss for reporting perplexity, which is the number people compare.
    ppl_criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    os.makedirs(CKPT_DIR, exist_ok=True)
    history, best_bleu, step = [], -1.0, 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch, tok_seen, loss_sum, n_batches = time.time(), 0, 0.0, 0
        # Gradient accumulation: run `accum_steps` minibatches before each optimizer
        # step, so the effective batch is accum_steps x max_tokens. The paper's WMT
        # runs use ~25k tokens per batch (Section 5.1); on this corpus 2,500 tokens
        # per forward pass x 4 gets to a comparable scale. Each minibatch's loss is
        # divided by accum_steps so the accumulated gradient is the mean over the
        # whole effective batch, not the sum.
        optimizer.zero_grad()
        pending = 0
        for src, tgt in batches_by_length(train_pairs, args.max_tokens):
            src, tgt = src.to(device), tgt.to(device)
            # Teacher forcing: input is tgt[:-1] (starts with BOS), target is tgt[1:].
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            (loss / args.accum_steps).backward()
            pending += 1
            tok_seen += (tgt[:, 1:] != PAD).sum().item()
            loss_sum += loss.item()
            n_batches += 1
            if pending < args.accum_steps:
                continue
            # Clip the accumulated gradient, not each minibatch's: the norm that
            # matters is the one actually applied to the weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            optimizer.zero_grad()
            pending = 0
            step += 1
            if step % args.log_every == 0:
                print(f"  step {step:5d}  loss {loss.item():.3f}  lr {scheduler.rate():.2e}  "
                      f"{tok_seen / (time.time() - t_epoch):,.0f} tok/s", flush=True)

        if pending:  # leftover minibatches at the end of the epoch
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        epoch_time = time.time() - t_epoch
        val_ppl = math.exp(eval_loss(model, val_pairs, ppl_criterion, device, args.max_tokens))
        val_bleu, _ = bleu_on(model, val_pairs, tgt_vocab, device, beam=0)
        rec = dict(epoch=epoch, step=step, train_loss=loss_sum / n_batches, val_ppl=val_ppl,
                   val_bleu_greedy=val_bleu, epoch_seconds=epoch_time,
                   tokens_per_sec=tok_seen / epoch_time)
        history.append(rec)
        print(f"epoch {epoch:2d} | train loss {rec['train_loss']:.3f} | val ppl {val_ppl:6.2f} "
              f"| val BLEU {val_bleu:5.2f} | {epoch_time:4.0f}s | {rec['tokens_per_sec']:,.0f} tok/s",
              flush=True)

        state = {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                 "val_bleu": val_bleu, "src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos}
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(state, ckpt_path(args.tokenizer, "best", args.variant))
            print(f"  saved best checkpoint (val BLEU {val_bleu:.2f})", flush=True)
        if args.keep_last > 0:
            # Per-epoch checkpoints for averaging (Section 6.1 averages the last 5).
            save_checkpoint(state, ckpt_path(args.tokenizer, f"epoch{epoch}", args.variant))
            stale = ckpt_path(args.tokenizer, f"epoch{epoch - args.keep_last}", args.variant)
            if os.path.exists(stale):
                os.remove(stale)

    total_time = time.time() - t_start
    print(f"training done in {total_time / 60:.1f} min; best val BLEU {best_bleu:.2f}")
    _save_results({"config": vars(args), "parameters": n_params, "device": str(device),
                   "train_minutes": total_time / 60, "history": history,
                   "best_val_bleu_greedy": best_bleu},
                  namespace=args.tokenizer + (f"_{args.variant}" if args.variant else ""))


def _save_results(update, namespace="word"):
    """Word-level results live at the top level (original layout); other tokenizers
    get their own sub-dict."""
    update_results(RESULTS, update, namespace=None if namespace == "word" else namespace)


def load_best(device, tokenizer="word", path=None):
    ckpt = load_checkpoint(path or ckpt_path(tokenizer, "best"), map_location=device)
    args = argparse.Namespace(**ckpt["args"])
    model = build_model(args, len(ckpt["src_itos"]), len(ckpt["tgt_itos"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate(args):
    device = get_device()
    model, ckpt = load_best(device, args.tokenizer, path=args.checkpoint)
    tok_args = ckpt["args"]
    _, _, test_pairs, src_vocab, tgt_vocab = load_multi30k(
        tokenizer=tok_args.get("tokenizer", "word"), bpe_merges=tok_args.get("bpe_merges", 8000))
    print(f"loaded epoch {ckpt['epoch']} checkpoint (val BLEU {ckpt['val_bleu']:.2f})")

    t0 = time.time()
    greedy_bleu, greedy_hyps = bleu_on(model, test_pairs, tgt_vocab, device, beam=0)
    t_greedy = time.time() - t0
    print(f"test BLEU greedy : {greedy_bleu:.2f}  ({t_greedy:.0f}s)")

    if args.beam == 0:  # greedy only; beam search over 1k sentences takes minutes on CPU
        _save_results({"test_bleu_greedy": greedy_bleu}, namespace=args.tokenizer)
        return
    t0 = time.time()
    beam_bleu, beam_hyps = bleu_on(model, test_pairs, tgt_vocab, device, beam=args.beam)
    t_beam = time.time() - t0
    print(f"test BLEU beam={args.beam}: {beam_bleu:.2f}  ({t_beam:.0f}s)")

    samples = []
    for i in [0, 1, 2, 3, 4, 100, 250, 500]:
        s, t = test_pairs[i]
        samples.append({"src": " ".join(to_words(src_vocab.decode(s))),
                        "ref": " ".join(to_words(tgt_vocab.decode(t))),
                        "greedy": " ".join(greedy_hyps[i]),
                        "beam": " ".join(beam_hyps[i])})
    for x in samples[:4]:
        print(f"\nDE   : {x['src']}\nREF  : {x['ref']}\nBEAM : {x['beam']}")

    _save_results({"test_bleu_greedy": greedy_bleu, "test_bleu_beam": beam_bleu,
                   "beam_size": args.beam, "samples": samples}, namespace=args.tokenizer)


def demo(args):
    device = get_device()
    model, ckpt = load_best(device, args.tokenizer)
    src_stoi = {w: i for i, w in enumerate(ckpt["src_itos"])}
    tgt_itos = ckpt["tgt_itos"]
    toks = tokenize(args.sentence)
    if ckpt["args"].get("tokenizer", "word") == "bpe":
        toks = BPE.load(os.path.join(DATA_DIR, f"bpe_{ckpt['args']['bpe_merges']}.json")).encode(toks)
    ids = [src_stoi.get(t, 3) for t in toks]
    out = model.beam_search(torch.tensor([ids], device=device), BOS, EOS, beam_size=args.beam)[0]
    words = to_words([tgt_itos[i] for i in out[1:] if i not in (PAD, EOS)])
    print(" ".join(words))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--d-ff", type=int, default=1024)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--max-tokens", type=int, default=4096)
    t.add_argument("--warmup", type=int, default=400)
    t.add_argument("--lr-factor", type=float, default=1.0)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tokenizer", choices=["word", "bpe"], default="word")
    t.add_argument("--bpe-merges", type=int, default=8000)
    t.add_argument("--keep-last", type=int, default=0,
                   help="keep per-epoch checkpoints for the last N epochs (for averaging)")
    t.add_argument("--pre-norm", action="store_true", help="x + Sublayer(LayerNorm(x)) layout")
    t.add_argument("--variant", default="", help="tag for checkpoint/results namespacing")
    t.add_argument("--accum-steps", type=int, default=1,
                   help="minibatches per optimizer step; effective batch = this x --max-tokens")

    e = sub.add_parser("evaluate")
    e.add_argument("--beam", type=int, default=4)
    e.add_argument("--tokenizer", choices=["word", "bpe"], default="word")
    e.add_argument("--checkpoint", default=None, help="explicit checkpoint path (e.g. an averaged one)")

    d = sub.add_parser("demo")
    d.add_argument("sentence")
    d.add_argument("--beam", type=int, default=4)
    d.add_argument("--tokenizer", choices=["word", "bpe"], default="word")

    args = p.parse_args()
    {"train": train, "evaluate": evaluate, "demo": demo}[args.cmd](args)
