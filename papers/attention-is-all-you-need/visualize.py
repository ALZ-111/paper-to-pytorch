"""
Figures for the README. Writes PNGs to ../../assets/attention/.

    python visualize.py            # all figures (needs checkpoints/multi30k_best.pt for attention maps)
    python visualize.py --no-model # only the figures that need no trained model
"""

import argparse
import json
import os

# torch must be imported before matplotlib: Anaconda ships two OpenMP runtimes
# (torch's libiomp5 and matplotlib's libomp) and loading them in the other order
# aborts the process.
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import BOS, EOS, PAD, tokenize
from model import PositionalEncoding

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "attention")
os.makedirs(OUT, exist_ok=True)


def plot_positional_encoding():
    pe = PositionalEncoding(128, dropout=0.0, max_len=100).pe[0]  # (100, 128)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    # Even dims are sin, odd are cos; plotting only the sin half avoids row striping.
    im = axes[0].imshow(pe[:, 0::2].T, aspect="auto", cmap="RdBu", origin="lower")
    axes[0].set_xlabel("position")
    axes[0].set_ylabel("sine dimension index i  (PE[pos, 2i])")
    axes[0].set_title("Sinusoidal positional encoding (d_model = 128)")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    for dim in [4, 5, 20, 21, 60, 61]:
        axes[1].plot(pe[:, dim], label=f"dim {dim}", lw=1.2)
    axes[1].set_xlabel("position")
    axes[1].set_title("Wavelength grows with dimension")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "positional_encoding.png"), dpi=130)
    plt.close(fig)


def plot_training_curves():
    path = os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        print("results.json not found; skipping training curves")
        return
    with open(path) as f:
        hist = json.load(f)["history"]
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(ep, [h["train_loss"] for h in hist], marker="o", ms=3)
    axes[0].set_title("Train loss (label-smoothed CE)")
    axes[1].plot(ep, [h["val_ppl"] for h in hist], marker="o", ms=3, color="C1")
    axes[1].set_yscale("log")
    axes[1].set_title("Validation perplexity")
    axes[2].plot(ep, [h["val_bleu_greedy"] for h in hist], marker="o", ms=3, color="C2")
    axes[2].set_title("Validation BLEU (greedy)")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"), dpi=130)
    plt.close(fig)


def _heatmap_grid(attn, xlabels, ylabels, title, fname, layer_names=None):
    """attn: (n_layers, n_heads, T, S)"""
    n_layers, n_heads = attn.shape[:2]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.1 * n_heads, 2.2 * n_layers), squeeze=False)
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            ax.imshow(attn[l, h], cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(xlabels)))
            ax.set_yticks(range(len(ylabels)))
            if l == n_layers - 1:
                ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
            else:
                ax.set_xticklabels([])
            if h == 0:
                ax.set_yticklabels(ylabels, fontsize=6)
                ax.set_ylabel(layer_names[l] if layer_names else f"layer {l + 1}", fontsize=8)
            else:
                ax.set_yticklabels([])
            if l == 0:
                ax.set_title(f"head {h + 1}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


def plot_attention_maps(sentence):
    from translate import load_best
    device = torch.device("cpu")
    model, ckpt = load_best(device)
    src_stoi = {w: i for i, w in enumerate(ckpt["src_itos"])}
    tgt_itos = ckpt["tgt_itos"]

    src_words = tokenize(sentence)
    src = torch.tensor([[src_stoi.get(w, 3) for w in src_words]])
    out_ids = model.beam_search(src, BOS, EOS, beam_size=4)[0]
    # Re-run a full teacher-forced pass so every layer's .attn is for the final output.
    tgt_in = torch.tensor([out_ids[:-1]]) if out_ids[-1] == EOS else torch.tensor([out_ids])
    with torch.no_grad():
        model(src, tgt_in)
    tgt_words = [tgt_itos[i] for i in tgt_in[0].tolist()]
    out_words = [tgt_itos[i] for i in out_ids[1:] if i != EOS]
    print("DE :", " ".join(src_words))
    print("EN :", " ".join(out_words))

    cross = torch.stack([l.cross_attn.attn[0] for l in model.decoder.layers])   # (L, H, T, S)
    enc_self = torch.stack([l.self_attn.attn[0] for l in model.encoder.layers])  # (L, H, S, S)
    dec_self = torch.stack([l.self_attn.attn[0] for l in model.decoder.layers])  # (L, H, T, T)

    _heatmap_grid(cross, src_words, tgt_words,
                  "Decoder cross-attention: which German word each English word reads",
                  "cross_attention.png")
    _heatmap_grid(enc_self, src_words, src_words,
                  "Encoder self-attention", "encoder_self_attention.png")
    _heatmap_grid(dec_self, tgt_words, tgt_words,
                  "Decoder self-attention (causal: upper triangle is masked)",
                  "decoder_self_attention.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-model", action="store_true")
    p.add_argument("--sentence", default="ein kleines mädchen in einem rosa kleid klettert auf eine treppe .")
    args = p.parse_args()
    plot_positional_encoding()
    plot_training_curves()
    if not args.no_model:
        plot_attention_maps(args.sentence)
    print("wrote figures to", os.path.abspath(OUT))
