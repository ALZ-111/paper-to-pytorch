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


def factor_comparison(results):
    """Ours vs the paper at 8 and 64 predictive factors, and the overfitting at 64."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # left: test HR@10 through training at 64 factors, where every model overfits
    for tag in ("gmf_f64", "mlp_f64", "neumf_f64", "neumf_f64_pretrain"):
        if tag not in results:
            continue
        h = results[tag]["history"]
        ax = axes[0]
        ax.plot([x["epoch"] for x in h], [x["hr"] for x in h], marker="o", ms=3, label=label(tag))
        peak = max(h[1:], key=lambda x: x["hr"])
        ax.plot([peak["epoch"]], [peak["hr"]], marker="*", ms=12, color=ax.lines[-1].get_color())
    axes[0].axhline(0.730, color="gray", ls="--", lw=1)
    axes[0].text(0.3, 0.7305, "paper NeuMF pre-trained 0.730", fontsize=7, color="gray", va="bottom")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("test HR@10"); axes[0].set_ylim(0.55, 0.75)
    axes[0].set_title("64 factors: every model peaks early, then declines")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    # right: NeuMF's best-epoch HR under each split, against the paper
    bars = [("neumf_f8", "8 factors\nour split", 0.688),
            ("neumf_f64", "64 factors\nour split", 0.705),
            ("neumf_f64_authors", "64 factors\nauthors' split", 0.705)]
    xs = range(len(bars))
    heights = [results[t]["best_on_test"]["hr"] if t in results else 0 for t, _, _ in bars]
    axes[1].bar(list(xs), heights, 0.6, color=["C0", "C0", "C2"])
    for x, (_, _, ref) in zip(xs, bars):
        axes[1].plot([x - 0.32, x + 0.32], [ref, ref], "k-", lw=2,
                     label="paper" if x == 0 else None)
    for x, h in zip(xs, heights):
        axes[1].text(x, h + 0.002, f"{h:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(list(xs)); axes[1].set_xticklabels([lbl for _, lbl, _ in bars], fontsize=8)
    axes[1].set_ylim(0.6, 0.74); axes[1].set_ylabel("test HR@10 (best epoch)")
    axes[1].set_title("Most of the 64-factor gap was the split, not the model")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    save_fig(fig, os.path.join(OUT, "factor_comparison.png"))


if __name__ == "__main__":
    training_curves(load_results(RESULTS))
    factor_comparison(load_results(RESULTS))
    print("wrote figures to", os.path.abspath(OUT))
