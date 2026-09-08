"""
Training script for Attention Is All You Need.

Toy task: reverse a sequence of integers. The encoder sees [3, 9, 4, 1] and the
decoder must emit [1, 4, 9, 3]. It is trivially generated, needs no dataset
download, and requires real cross-attention to solve (the decoder must look up
position S-1-i in the source), so it is a good end-to-end check that the
architecture, masking and shifted-target loss are wired correctly.

Run:  python train.py
"""

import os
import time

import torch
import torch.nn as nn

from model import Transformer

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "toy_reverse.pt")

PAD, BOS, EOS = 0, 1, 2
VOCAB = 20          # tokens 3..19 are "content" tokens
SEQ_LEN = 10        # content length (before BOS/EOS)


def make_batch(batch_size, device):
    """Returns src (B, S), tgt_in (B, T), tgt_out (B, T)."""
    content = torch.randint(3, VOCAB, (batch_size, SEQ_LEN), device=device)
    reversed_ = content.flip(1)
    bos = torch.full((batch_size, 1), BOS, device=device)
    eos = torch.full((batch_size, 1), EOS, device=device)
    src = content
    # Teacher forcing: decoder input is the target shifted right (starts with BOS),
    # the loss target is the target shifted left (ends with EOS).
    tgt_in = torch.cat([bos, reversed_], dim=1)   # [BOS, r1, ..., rS]
    tgt_out = torch.cat([reversed_, eos], dim=1)  # [r1, ..., rS, EOS]
    return src, tgt_in, tgt_out


class NoamLR:
    """Learning-rate schedule from Section 5.3:
        lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    Linear warmup, then inverse-square-root decay."""

    def __init__(self, optimizer, d_model, warmup=400, factor=1.0):
        self.opt, self.d_model, self.warmup, self.factor = optimizer, d_model, warmup, factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.factor * self.d_model ** -0.5 * min(
            self.step_num ** -0.5, self.step_num * self.warmup ** -1.5
        )
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        return lr


def train(model, device, steps=600, batch_size=64, log_every=50):
    model.to(device).train()
    # Adam hyperparameters from Section 5.3.
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, model.encoder.d_model, warmup=200)
    # Label smoothing 0.1 as in the paper; ignore padding positions.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    t0 = time.time()
    for step in range(1, steps + 1):
        src, tgt_in, tgt_out = make_batch(batch_size, device)
        logits = model(src, tgt_in)                               # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        lr = scheduler.step()

        if step % log_every == 0:
            acc = evaluate(model, device)
            model.train()
            print(f"step {step:4d} | loss {loss.item():.3f} | lr {lr:.2e} "
                  f"| seq acc {acc:.1%} | {time.time() - t0:.0f}s")


@torch.no_grad()
def evaluate(model, device, n=256):
    """Exact-match accuracy over whole sequences using greedy decoding."""
    src, _, tgt_out = make_batch(n, device)
    pred = model.greedy_decode(src, BOS, EOS, max_len=SEQ_LEN + 2)[:, 1:]  # drop BOS
    pred = pred[:, : SEQ_LEN + 1]
    return (pred == tgt_out).all(dim=1).float().mean().item()


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    torch.manual_seed(0)

    # Small model: the task is easy, and this trains in about a minute on CPU.
    model = Transformer(VOCAB, VOCAB, d_model=128, n_layers=2, n_heads=4, d_ff=512, dropout=0.1)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, device)

    # Persist the learned weights. state_dict() is just a dict of tensors keyed by
    # parameter name; the architecture itself (model.py) is not stored, so loading
    # requires constructing the same Transformer(...) first.
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"saved weights to {CKPT_PATH}")

    # Prove the round trip works: build a fresh model, load the weights, decode.
    restored = Transformer(VOCAB, VOCAB, d_model=128, n_layers=2, n_heads=4, d_ff=512, dropout=0.1)
    restored.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    restored.to(device)
    src, _, _ = make_batch(3, device)
    out = restored.greedy_decode(src, BOS, EOS, max_len=SEQ_LEN + 2)
    for s, o in zip(src.tolist(), out.tolist()):
        print(f"src {s}\n -> {o[1:]}")
