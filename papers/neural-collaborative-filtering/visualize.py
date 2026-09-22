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

    # right: best-epoch HR at 8 vs 64 factors, ours against the paper
    paper = {"neumf": {8: 0.688, 64: 0.705}, "neumf_pretrain": {8: 0.684, 64: 0.730}}
    width, xs = 0.35, [0, 1]
    for ax_i, (key, name) in enumerate([("neumf", "NeuMF"), ("neumf_pretrain", "NeuMF pre-trained")]):
        ours = []
        for f in (8, 64):
            tag = f"neumf_f{f}" + ("_pretrain" if key.endswith("pretrain") else "")
            ours.append(results[tag]["best_on_test"]["hr"] if tag in results else 0)
        off = (ax_i - 0.5) * width
        axes[1].bar([x + off for x in xs], ours, width * 0.9, label=f"{name}, ours")
        axes[1].plot([x + off for x in xs], [paper[key][f] for f in (8, 64)], "k_",
                     ms=14, mew=2, label=f"{name}, paper" if ax_i == 0 else None)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(["8 factors", "64 factors"])
    axes[1].set_ylim(0.6, 0.76); axes[1].set_ylabel("test HR@10 (best epoch)")
    axes[1].set_title("Matches the paper at 8 factors, not at 64")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, axis="y")
    save_fig(fig, os.path.join(OUT, "factor_comparison.png"))


if __name__ == "__main__":
    training_curves(load_results(RESULTS))
    factor_comparison(load_results(RESULTS))
    print("wrote figures to", os.path.abspath(OUT))
