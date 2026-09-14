"""
How many latent dimensions does each model actually use?

Two complementary diagnostics, computed on the MNIST test set for every checkpoint:

  * Active units (Burda, Grosse & Salakhutdinov, 2015): dimension u is active if
        Var_x( E_{q(z|x)}[z_u] ) > 0.01
    i.e. the posterior mean of that dimension moves as the input changes.

  * Per-dimension KL: KL_u = E_x[ KL( q(z_u|x) || N(0,1) ) ]. A dimension the decoder
    ignores is driven to the prior (mu = 0, sigma = 1) and contributes ~0 nats.

Together they show why Z = 200 does not overfit: most of its dimensions are switched off.

    python active_units.py        # prints a table, writes results.json["active_units"]
                                  # and ../../assets/vae/active_units.png
"""

import json
import os

import torch  # before matplotlib: OpenMP runtime clash on Anaconda otherwise
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import kl_standard_normal
from train import RESULTS, load_checkpoint, load_mnist

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "vae")


@torch.no_grad()
def analyse(model, x, threshold=0.01):
    mu, logvar = model.encoder(x)
    mean_var = mu.var(dim=0)                                        # (Z,)
    per_dim_kl = torch.stack(
        [kl_standard_normal(mu[:, [u]], logvar[:, [u]]).mean() for u in range(mu.size(1))])
    return {
        "active_units": int((mean_var > threshold).sum()),
        "z_dim": mu.size(1),
        "posterior_mean_variance": mean_var.sort(descending=True).values.tolist(),
        "per_dim_kl": per_dim_kl.sort(descending=True).values.tolist(),
        "total_kl": float(per_dim_kl.sum()),
    }


def main():
    _, x_test, _ = load_mnist("cpu")
    with open(RESULTS) as f:
        runs = json.load(f)
    keys = sorted(runs, key=lambda k: (int(k[1:].split("_")[0]), k))

    print(f"{'run':>12} {'Z':>4} {'active':>7} {'total KL':>9}  {'top-5 per-dim KL (nats)'}")
    report = {}
    for key in keys:
        model, _ = load_checkpoint(key)
        r = analyse(model, x_test)
        report[key] = r
        top = " ".join(f"{v:5.2f}" for v in r["per_dim_kl"][:5])
        print(f"{key:>12} {r['z_dim']:>4} {r['active_units']:>7} {r['total_kl']:>9.2f}  {top}")
        runs[key]["active_units"] = {"count": r["active_units"], "total_kl": r["total_kl"]}

    with open(RESULTS, "w") as f:
        json.dump(runs, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in keys:
        if "_" in key:
            continue  # MLP/ELBO runs only, one curve per Z
        r = report[key]
        axes[0].plot(range(1, r["z_dim"] + 1), r["per_dim_kl"], marker=".", ms=3,
                     label=f"Z = {r['z_dim']}  ({r['active_units']} active)")
    axes[0].set_xscale("log"); axes[0].set_xlabel("latent dimension (sorted by KL)")
    axes[0].set_ylabel("KL to prior (nats)"); axes[0].set_title("Per-dimension KL: unused dimensions sit at zero")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    zs = [report[k]["z_dim"] for k in keys if "_" not in k]
    act = [report[k]["active_units"] for k in keys if "_" not in k]
    axes[1].plot(zs, act, marker="o"); axes[1].plot(zs, zs, ls="--", color="gray", label="all dimensions")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Z"); axes[1].set_ylabel("active units")
    axes[1].set_title("Active units plateau far below Z"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "active_units.png"), dpi=130)
    print("wrote", os.path.join(os.path.abspath(OUT), "active_units.png"))


if __name__ == "__main__":
    main()
