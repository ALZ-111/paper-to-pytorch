"""
How trustworthy is the reported log p(x)?

The importance-sampled estimator (Appendix D of the paper) is what everyone quotes, but
a single number hides two things:

  * **Bias.** log (1/L) sum_i w_i is a *lower* bound on log p(x) in expectation, by
    Jensen's inequality, and it rises monotonically with L. Any reported figure is
    therefore conservative, and two papers using different L are not comparable.
  * **Variance.** With finite L the estimate wobbles from seed to seed.

This script measures both: log p(x) across sample counts and seeds, on a fixed subset.
It also tests antithetic sampling (drawing L/2 noise vectors and using each with both
signs, which is valid because q is symmetric about mu) as a variance-reduction trick.

    python variance_study.py [--n-images 200] [--seeds 20] [--runs z20 z20_conv]

Writes results.json["variance_study"] and ../../assets/vae/variance_study.png.
"""

import argparse
import math
import os

import torch

import _bootstrap  # noqa: F401
from utils.plotting import plt, save_fig
from utils.results import update_results
from model import bernoulli_log_likelihood_many
from train import RESULTS, load_checkpoint, load_mnist

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "vae")


@torch.inference_mode()
def log_px(model, x, n_samples, antithetic=False, seed=0, chunk=250):
    """One importance-sampled estimate of mean log p(x) over `x`, at a given seed."""
    torch.manual_seed(seed)
    mu, logvar = model.encoder(x)
    std = torch.exp(0.5 * logvar)
    parts = []
    done = 0
    while done < n_samples:
        k = min(chunk, n_samples - done)
        if antithetic:
            # k must be even for pairing; the last chunk may be odd, so pad by one.
            half = torch.randn((k + 1) // 2, *mu.shape, device=x.device)
            eps = torch.cat([half, -half])[:k]
        else:
            eps = torch.randn(k, *mu.shape, device=x.device)
        z = mu + std * eps
        logits = model.decoder(z.view(-1, model.z_dim)).view(k, x.size(0), -1)
        parts.append(bernoulli_log_likelihood_many(logits, x)
                     - 0.5 * (z.pow(2) - eps.pow(2)).sum(-1) + 0.5 * logvar.sum(-1))
        done += k
    log_w = torch.cat(parts, dim=0)
    return (torch.logsumexp(log_w, dim=0) - math.log(n_samples)).mean().item()


def mean_sd(values):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    return m, (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def main(args):
    _, x_test, _ = load_mnist("cpu")
    x = x_test[: args.n_images]
    report = {}

    for run in args.runs:
        model, _ = load_checkpoint(run)
        rows = []
        print(f"\n{run}: mean log p(x) over {args.n_images} test images, {args.seeds} seeds each")
        print(f"{'L':>6} {'plain':>18} {'antithetic':>18}")
        for n in args.samples:
            p_m, p_sd = mean_sd([log_px(model, x, n, False, s) for s in range(args.seeds)])
            a_m, a_sd = mean_sd([log_px(model, x, n, True, s) for s in range(args.seeds)])
            rows.append({"n_samples": n, "plain_mean": p_m, "plain_sd": p_sd,
                         "antithetic_mean": a_m, "antithetic_sd": a_sd,
                         "variance_ratio": (p_sd / a_sd) ** 2 if a_sd else float("nan")})
            print(f"{n:>6} {p_m:>11.3f} ± {p_sd:.3f} {a_m:>11.3f} ± {a_sd:.3f}", flush=True)
        report[run] = rows

    update_results(RESULTS, {"n_images": args.n_images, "seeds": args.seeds, "runs": report},
                   namespace="variance_study")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for run, rows in report.items():
        ns = [r["n_samples"] for r in rows]
        axes[0].errorbar(ns, [r["plain_mean"] for r in rows], yerr=[r["plain_sd"] for r in rows],
                         marker="o", capsize=3, label=run)
        axes[1].plot(ns, [r["plain_sd"] for r in rows], marker="o", label=f"{run} plain")
        axes[1].plot(ns, [r["antithetic_sd"] for r in rows], marker="s", ls="--",
                     label=f"{run} antithetic")
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("importance samples L")
    axes[0].set_ylabel("estimated log p(x)")
    axes[0].set_title("The estimate is a lower bound that rises with L")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
    axes[1].set_xlabel("importance samples L"); axes[1].set_ylabel("sd across seeds (nats)")
    axes[1].set_title("Antithetic sampling does not reduce the spread")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    save_fig(fig, os.path.join(OUT, "variance_study.png"))
    print("\nwrote", os.path.join(os.path.abspath(OUT), "variance_study.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--samples", type=int, nargs="+", default=[32, 128, 512, 2048])
    p.add_argument("--runs", nargs="+", default=["z20", "z20_conv"])
    main(p.parse_args())
