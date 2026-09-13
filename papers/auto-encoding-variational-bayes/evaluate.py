"""
The paper's evaluation: ELBO and importance-sampled marginal likelihood on the MNIST
test set for every trained latent size (Section 5, Figures 2 and 3).

    python evaluate.py [--samples 5000] [--n-test 10000]

Writes the numbers into results.json under each run's "eval" key and prints a table.
"""

import argparse
import json
import time

import torch

from train import RESULTS, load_checkpoint, load_mnist


@torch.no_grad()
def evaluate_run(model, x, n_samples, batch_size):
    elbo_sum, iw_sum, n = 0.0, 0.0, 0
    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size]
        elbo, _, _ = model.elbo(xb, n_samples=16)          # low-variance ELBO estimate
        iw = model.log_marginal_likelihood(xb, n_samples=n_samples, chunk=250)
        elbo_sum += elbo.sum().item()
        iw_sum += iw.sum().item()
        n += xb.size(0)
    return elbo_sum / n, iw_sum / n


def main(args):
    torch.manual_seed(0)
    _, x_test, _ = load_mnist("cpu")
    x_test = x_test[: args.n_test]
    with open(RESULTS) as f:
        runs = json.load(f)

    print(f"{'Z':>4} {'test ELBO':>11} {'log p(x) (IW)':>15} {'gap':>6} {'time':>6}")
    for key in sorted(runs, key=lambda k: int(k[1:])):
        z = int(key[1:])
        model, _ = load_checkpoint(z)
        t0 = time.time()
        elbo, logpx = evaluate_run(model, x_test, args.samples, batch_size=100)
        runs[key]["eval"] = {"test_elbo": elbo, "test_log_px_iw": logpx,
                             "iw_samples": args.samples, "n_test": args.n_test}
        print(f"{z:>4} {elbo:>11.2f} {logpx:>15.2f} {logpx - elbo:>6.2f} {time.time() - t0:>5.0f}s", flush=True)

    with open(RESULTS, "w") as f:
        json.dump(runs, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=5000, help="importance samples per image")
    p.add_argument("--n-test", type=int, default=10000)
    main(p.parse_args())
