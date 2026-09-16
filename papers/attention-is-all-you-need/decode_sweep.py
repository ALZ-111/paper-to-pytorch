"""
How much do beam size and the length penalty matter? (Section 6.1 of the paper picks
beam 4 and alpha = 0.6 on the dev set.)

    python decode_sweep.py [--tokenizer bpe --checkpoint checkpoints/multi30k_bpe_avg5.pt]

Decodes the test set for every (beam, alpha) pair, prints BLEU and mean output length,
saves results.json["decode_sweep"] and ../../assets/attention/decode_sweep.png.
"""

import argparse
import json
import os
import time

import torch

import _bootstrap  # noqa: F401
from utils.plotting import plt, save_fig
from utils.results import update_results
from bleu import corpus_bleu
from data import BOS, EOS, collate, load_multi30k, to_words
from translate import RESULTS, get_device, load_best

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "attention")


@torch.no_grad()
def decode_all(model, pairs, device, beam, alpha, batch_size=128):
    hyps = []
    for i in range(0, len(pairs), batch_size):
        src, _ = collate(pairs[i : i + batch_size])
        src = src.to(device)
        if beam == 1:
            outs = model.greedy_decode(src, BOS, EOS, max_len=60).tolist()
        else:
            outs = model.beam_search(src, BOS, EOS, beam_size=beam, max_len=60, length_penalty=alpha)
        hyps.extend(outs)
    return hyps


def main(args):
    device = get_device()
    model, ckpt = load_best(device, args.tokenizer, path=args.checkpoint)
    tok_args = ckpt["args"]
    _, _, test, _, tgt_vocab = load_multi30k(tokenizer=tok_args.get("tokenizer", "word"),
                                             bpe_merges=tok_args.get("bpe_merges", 8000))
    refs = [to_words(tgt_vocab.decode(t)) for _, t in test]
    ref_len = sum(len(r) for r in refs) / len(refs)

    rows = []
    print(f"reference mean length {ref_len:.2f} words")
    print(f"{'beam':>5} {'alpha':>6} {'BLEU':>7} {'len':>6} {'len/ref':>8} {'time':>6}")
    for beam in args.beams:
        for alpha in (args.alphas if beam > 1 else [0.0]):
            t0 = time.time()
            hyps = [to_words(tgt_vocab.decode(h)) for h in decode_all(model, test, device, beam, alpha)]
            bleu = corpus_bleu(hyps, refs)
            hyp_len = sum(len(h) for h in hyps) / len(hyps)
            rows.append({"beam": beam, "alpha": alpha, "bleu": bleu, "mean_len": hyp_len,
                         "len_ratio": hyp_len / ref_len, "seconds": time.time() - t0})
            print(f"{beam:>5} {alpha:>6.1f} {bleu:>7.2f} {hyp_len:>6.2f} {hyp_len / ref_len:>8.3f} "
                  f"{rows[-1]['seconds']:>5.0f}s", flush=True)

    update_results(RESULTS, {"checkpoint": args.checkpoint or "best", "tokenizer": args.tokenizer,
                             "reference_mean_len": ref_len, "rows": rows}, namespace="decode_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for alpha in args.alphas:
        pts = [r for r in rows if r["alpha"] == alpha or r["beam"] == 1]
        pts = sorted({r["beam"]: r for r in pts}.values(), key=lambda r: r["beam"])
        axes[0].plot([r["beam"] for r in pts], [r["bleu"] for r in pts], marker="o", label=f"α = {alpha}")
        axes[1].plot([r["beam"] for r in pts], [r["len_ratio"] for r in pts], marker="o", label=f"α = {alpha}")
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("beam size"); axes[0].set_ylabel("test BLEU")
    axes[0].set_title("BLEU vs beam size and length penalty"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("beam size"); axes[1].set_ylabel("hyp length / ref length")
    axes[1].set_title("Larger beams prefer shorter outputs unless penalised"); axes[1].legend(); axes[1].grid(alpha=0.3)
    save_fig(fig, os.path.join(OUT, "decode_sweep.png"))
    print("wrote", os.path.join(os.path.abspath(OUT), "decode_sweep.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="bpe")
    p.add_argument("--checkpoint", default="checkpoints/multi30k_bpe_avg5.pt")
    p.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.6, 1.0])
    main(p.parse_args())
