"""
Figures for the README. Writes PNGs to ../../assets/vae/.

    python visualize.py          # needs checkpoints/vae_z20.pt and vae_z2.pt
"""

import json
import os

import torch  # before matplotlib: OpenMP runtime clash on Anaconda otherwise
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from train import load_checkpoint, load_mnist, RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "assets", "vae")
os.makedirs(OUT, exist_ok=True)


def grid(images, n_cols, path, title=None, scale=1.0):
    """images: (N, 784) probabilities -> N/n_cols rows of 28x28 tiles."""
    n = images.size(0)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].view(28, 28), cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def reconstructions(model, x_test):
    """Top row originals, bottom row posterior-mean reconstructions."""
    x = x_test[:16]
    grid(torch.cat([x, model.reconstruct(x)]), 16,
         os.path.join(OUT, "reconstructions.png"),
         "Top: test images. Bottom: reconstructions through a 20-d latent", scale=0.8)


def prior_samples(model):
    torch.manual_seed(0)
    grid(model.sample(100), 10, os.path.join(OUT, "samples.png"),
         "100 samples from the prior, z ~ N(0, I), decoded", scale=0.6)


def latent_manifold(model2):
    """Paper Figure 4: walk a 2-d latent along the inverse Gaussian CDF so the grid
    covers the prior's mass evenly, and decode each point."""
    n = 20
    lin = torch.tensor(norm.ppf(torch.linspace(0.02, 0.98, n)), dtype=torch.float)
    zs = torch.stack(torch.meshgrid(lin, lin.flip(0), indexing="xy"), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        imgs = torch.sigmoid(model2.decoder(zs))
    grid(imgs, n, os.path.join(OUT, "latent_manifold.png"),
         "Learned 2-d manifold: decoded z on a grid of prior quantiles", scale=0.45)


def latent_scatter(model2, x_test, y_test):
    with torch.no_grad():
        mu, _ = model2.encoder(x_test)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap="tab10", s=3, alpha=0.7)
    ax.set_title("Posterior means μ(x) of 10k test digits in the 2-d latent space")
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latent_scatter.png"), dpi=130)
    plt.close(fig)


def training_curves():
    with open(RESULTS) as f:
        runs = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for key in sorted(runs, key=lambda k: int(k[1:])):
        h = runs[key]["history"]
        axes[0].plot([r["epoch"] for r in h], [r["test_elbo"] for r in h], label=f"Z = {key[1:]}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("test ELBO (nats)")
    axes[0].set_title("ELBO vs latent dimension (cf. paper Fig. 2)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    if "z20" in runs:
        h = runs["z20"]["history"]
        ep = [r["epoch"] for r in h]
        axes[1].plot(ep, [-r["test_recon"] for r in h], label="−E[log p(x|z)] (reconstruction)")
        axes[1].plot(ep, [r["test_kl"] for r in h], label="KL(q‖p)")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("nats"); axes[1].set_yscale("log")
        axes[1].set_title("Z = 20: the two terms of −ELBO"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    _, x_test, y_test = load_mnist("cpu")
    model20, _ = load_checkpoint(20)
    model2, _ = load_checkpoint(2)
    reconstructions(model20, x_test)
    prior_samples(model20)
    latent_manifold(model2)
    latent_scatter(model2, x_test, y_test)
    training_curves()
    print("wrote figures to", os.path.abspath(OUT))
