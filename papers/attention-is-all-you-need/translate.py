"""
Train and evaluate the Transformer on Multi30k German -> English.

    python translate.py train    [--epochs 15] [--d-model 256] ...
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

from bleu import corpus_bleu
from data import BOS, EOS, PAD, batches_by_length, collate, load_multi30k, tokenize
from model import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
BEST_CKPT = os.path.join(CKPT_DIR, "multi30k_best.pt")
RESULTS = os.path.join(HERE, "results.json")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    )


@torch.no_grad()
def translate_corpus(model, pairs, tgt_vocab, device, beam=0, batch_size=128, max_len=60):
    """Decode every source sentence in `pairs`. Returns list of token lists."""
    model.eval()
    hyps = []
    if beam > 0:
        for src_ids, _ in pairs:
            src = torch.tensor([src_ids], device=device)
            out = model.beam_search(src, BOS, EOS, beam_size=beam, max_len=max_len)
            hyps.append(tgt_vocab.decode(out))
    else:
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            src, _ = collate(chunk)
            out = model.greedy_decode(src.to(device), BOS, EOS, max_len=max_len)
            hyps.extend(tgt_vocab.decode(row) for row in out.tolist())
    return hyps


def bleu_on(model, pairs, tgt_vocab, device, beam=0):
    hyps = translate_corpus(model, pairs, tgt_vocab, device, beam=beam)
    refs = [tgt_vocab.decode(t) for _, t in pairs]
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
    torch.manual_seed(args.seed)
    print(f"device: {device}  threads: {torch.get_num_threads()}")

    train_pairs, val_pairs, test_pairs, src_vocab, tgt_vocab = load_multi30k()
    print(f"train {len(train_pairs)}  val {len(val_pairs)}  test {len(test_pairs)}  "
          f"src vocab {len(src_vocab)}  tgt vocab {len(tgt_vocab)}")

    model = build_model(args, len(src_vocab), len(tgt_vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
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
        for src, tgt in batches_by_length(train_pairs, args.max_tokens):
            src, tgt = src.to(device), tgt.to(device)
            # Teacher forcing: input is tgt[:-1] (starts with BOS), target is tgt[1:].
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            step += 1
            tok_seen += (tgt[:, 1:] != PAD).sum().item()
            loss_sum += loss.item()
            n_batches += 1
            if step % args.log_every == 0:
                print(f"  step {step:5d}  loss {loss.item():.3f}  lr {scheduler.rate():.2e}  "
                      f"{tok_seen / (time.time() - t_epoch):,.0f} tok/s", flush=True)

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

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "val_bleu": val_bleu, "src_itos": src_vocab.itos,
                        "tgt_itos": tgt_vocab.itos}, BEST_CKPT)
            print(f"  saved best checkpoint (val BLEU {val_bleu:.2f})", flush=True)

    total_time = time.time() - t_start
    print(f"training done in {total_time / 60:.1f} min; best val BLEU {best_bleu:.2f}")
    _save_results({"config": vars(args), "parameters": n_params, "device": str(device),
                   "train_minutes": total_time / 60, "history": history,
                   "best_val_bleu_greedy": best_bleu})


def _save_results(update):
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    data.update(update)
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)


def load_best(device):
    ckpt = torch.load(BEST_CKPT, map_location=device)
    args = argparse.Namespace(**ckpt["args"])
    model = build_model(args, len(ckpt["src_itos"]), len(ckpt["tgt_itos"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def evaluate(args):
    device = get_device()
    model, ckpt = load_best(device)
    _, _, test_pairs, src_vocab, tgt_vocab = load_multi30k()
    print(f"loaded epoch {ckpt['epoch']} checkpoint (val BLEU {ckpt['val_bleu']:.2f})")

    t0 = time.time()
    greedy_bleu, greedy_hyps = bleu_on(model, test_pairs, tgt_vocab, device, beam=0)
    t_greedy = time.time() - t0
    print(f"test BLEU greedy : {greedy_bleu:.2f}  ({t_greedy:.0f}s)")

    t0 = time.time()
    beam_bleu, beam_hyps = bleu_on(model, test_pairs, tgt_vocab, device, beam=args.beam)
    t_beam = time.time() - t0
    print(f"test BLEU beam={args.beam}: {beam_bleu:.2f}  ({t_beam:.0f}s)")

    samples = []
    for i in [0, 1, 2, 3, 4, 100, 250, 500]:
        s, t = test_pairs[i]
        samples.append({"src": " ".join(src_vocab.decode(s)),
                        "ref": " ".join(tgt_vocab.decode(t)),
                        "greedy": " ".join(greedy_hyps[i]),
                        "beam": " ".join(beam_hyps[i])})
    for x in samples[:4]:
        print(f"\nDE   : {x['src']}\nREF  : {x['ref']}\nBEAM : {x['beam']}")

    _save_results({"test_bleu_greedy": greedy_bleu, "test_bleu_beam": beam_bleu,
                   "beam_size": args.beam, "samples": samples})


def demo(args):
    device = get_device()
    model, ckpt = load_best(device)
    src_stoi = {w: i for i, w in enumerate(ckpt["src_itos"])}
    tgt_itos = ckpt["tgt_itos"]
    ids = [src_stoi.get(t, 3) for t in tokenize(args.sentence)]
    out = model.beam_search(torch.tensor([ids], device=device), BOS, EOS, beam_size=args.beam)
    words = [tgt_itos[i] for i in out[1:] if i not in (PAD, EOS)]
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

    e = sub.add_parser("evaluate")
    e.add_argument("--beam", type=int, default=4)

    d = sub.add_parser("demo")
    d.add_argument("sentence")
    d.add_argument("--beam", type=int, default=4)

    args = p.parse_args()
    {"train": train, "evaluate": evaluate, "demo": demo}[args.cmd](args)
