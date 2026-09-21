"""
Figures for the README. Writes PNGs to ../../assets/ncf/.

    python visualize.py
"""

import os

import torch  # noqa: F401  (before numpy-heavy imports; see utils/plotting.py)

import _bootstrap  # noqa: F401
from utils.plotting import plt, save_fig
from utils.results import load_results

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "ncf")
RESULTS = os.path.join(HERE, "results.json")

LABELS = {"gmf": "GMF", "mlp": "MLP", "neumf": "NeuMF", "neumf_pretrain": "NeuMF (pre-trained)"}


def label(tag):
    kind = tag.rsplit("_f", 1)[0]
    rest = tag.split("_f", 1)[1]
    pre = rest.endswith("_pretrain")
    return LABELS["neumf_pretrain" if pre else kind]


def training_curves(results, factors=8):
    runs = {t: r for t, r in results.items() if f"_f{factors}" in t}
    if not runs:
        print("no runs yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for tag, r in runs.items():
        h = r["history"]
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["hr"] for x in h], marker="o", ms=3, label=label(tag))
        axes[1].plot(ep, [x["ndcg"] for x in h], marker="o", ms=3, label=label(tag))
    paper = {"HR@10": (0.688, 0.684), "NDCG@10": (0.410, 0.403)}   # Table 2, factors 8
    for ax, (name, (scratch, pre)) in zip(axes, paper.items()):
        ax.axhline(scratch, color="gray", ls="--", lw=1)
        ax.text(0.2, scratch, f"paper NeuMF {scratch}", va="bottom", fontsize=7, color="gray")
        ax.set_xlabel("epoch"); ax.set_ylabel(name); ax.grid(alpha=0.3)
        ax.set_title(f"Test {name}, {factors} predictive factors")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.35, None); axes[1].set_ylim(0.18, None)
    save_fig(fig, os.path.join(OUT, "training_curves.png"))


if __name__ == "__main__":
    training_curves(load_results(RESULTS))
    print("wrote figures to", os.path.abspath(OUT))
