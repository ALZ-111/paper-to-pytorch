"""
Train the VAE on binarised MNIST and track the ELBO.

    python train.py --z-dim 20 --epochs 30          # main model
    python train.py --z-dim 2  --epochs 30          # for the latent-manifold figure
    python train.py --z-dim 20 --arch conv          # convolutional encoder/decoder
    python train.py --z-dim 20 --iwae-k 5           # importance-weighted objective
    python train.py --z-dim 20 --beta 4             # beta-VAE
    python train.py --z-dim 20 --kl-warmup 10       # anneal beta 0 -> 1 over 10 epochs

Binarisation: the paper models MNIST pixels as Bernoulli. We use the common
"dynamic binarisation" scheme, sampling x ~ Bernoulli(pixel intensity) afresh for every
minibatch, which acts as data augmentation and avoids overfitting a fixed binarised set.
Evaluation uses the fixed intensities as Bernoulli targets (the standard practice).

Writes checkpoints/vae_{tag}.pt and appends a run record to results.json, where tag is
z{Z} for the paper's MLP/ELBO setup, with _conv and/or _iwae{k} suffixes for variants.
"""

import argparse
import json
import os
import time

import torch
from torchvision import datasets

from model import VAE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULTS = os.path.join(HERE, "results.json")


def load_mnist(device):
    """Returns train (60000, 784) and test (10000, 784) float tensors in [0, 1],
    plus test labels, all resident in memory (MNIST is 47 MB as float32)."""
    tr = datasets.MNIST(DATA, train=True, download=True)
    te = datasets.MNIST(DATA, train=False, download=True)
    x_train = tr.data.view(-1, 784).float().div(255).to(device)
    x_test = te.data.view(-1, 784).float().div(255).to(device)
    return x_train, x_test, te.targets.to(device)


@torch.no_grad()
def evaluate(model, x, batch_size=1000):
    """Mean ELBO, reconstruction term and KL over a dataset, using pixel intensities
    as soft Bernoulli targets and 1 posterior sample per image."""
    model.eval()
    tot = torch.zeros(3, device=x.device)
    for i in range(0, x.size(0), batch_size):
        elbo, recon, kl = model.elbo(x[i : i + batch_size])
        tot += torch.stack([elbo.sum(), recon.sum(), kl.sum()])
    return (tot / x.size(0)).tolist()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    x_train, x_test, _ = load_mnist(device)

    model = VAE(784, args.h_dim, args.z_dim, arch=args.arch).to(device)
    tag = run_tag(args.z_dim, args.arch, args.iwae_k, args.beta, args.kl_warmup)
    n_params = sum(p.numel() for p in model.parameters())
    # The paper used Adagrad; Adam (same first author, one year later) is the modern default.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"run={tag}  params={n_params:,}  device={device}")

    history, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(x_train.size(0), device=device)
        epoch_elbo, n_batches = 0.0, 0
        # KL warm-up (Bowman et al. 2016 / Sonderby et al. 2016): ramp the KL weight
        # from 0 to beta over the first kl_warmup epochs so the decoder learns to use
        # z before the KL term can push the posterior onto the prior.
        beta = args.beta * min(1.0, epoch / args.kl_warmup) if args.kl_warmup > 0 else args.beta
        for i in range(0, x_train.size(0), args.batch_size):
            x = x_train[perm[i : i + args.batch_size]]
            x = torch.bernoulli(x)  # dynamic binarisation
            if args.iwae_k > 1:
                bound = model.iwae(x, k=args.iwae_k)
            else:
                bound, _, _ = model.elbo(x, beta=beta)
            loss = -bound.mean()     # maximise the bound = minimise its negative
            elbo = bound
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_elbo += elbo.mean().item()
            n_batches += 1

        test_elbo, test_recon, test_kl = evaluate(model, x_test)
        rec = dict(epoch=epoch, train_elbo=epoch_elbo / n_batches, test_elbo=test_elbo,
                   test_recon=test_recon, test_kl=test_kl, seconds=time.time() - t0)
        history.append(rec)
        print(f"epoch {epoch:3d} | train ELBO {rec['train_elbo']:8.2f} | test ELBO {test_elbo:8.2f} "
              f"(recon {test_recon:8.2f}, KL {test_kl:6.2f}) | {rec['seconds']:5.0f}s", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"vae_{tag}.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, ckpt_path)
    print(f"saved {ckpt_path}")

    runs = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            runs = json.load(f)
    runs[tag] = {"config": vars(args), "parameters": n_params,
                              "final_test_elbo": history[-1]["test_elbo"],
                              "final_test_kl": history[-1]["test_kl"],
                              "minutes": (time.time() - t0) / 60, "history": history}
    with open(RESULTS, "w") as f:
        json.dump(runs, f, indent=2)


def run_tag(z_dim, arch="mlp", iwae_k=1, beta=1.0, kl_warmup=0):
    tag = f"z{z_dim}"
    if arch != "mlp":
        tag += f"_{arch}"
    if iwae_k > 1:
        tag += f"_iwae{iwae_k}"
    if beta != 1.0:
        tag += f"_beta{beta:g}"
    if kl_warmup > 0:
        tag += f"_warmup{kl_warmup}"
    return tag


def load_checkpoint(tag, device="cpu"):
    """tag: e.g. 20, "z20", "z20_conv", "z20_iwae5"."""
    if isinstance(tag, int):
        tag = f"z{tag}"
    ckpt = torch.load(os.path.join(CKPT_DIR, f"vae_{tag}.pt"), map_location=device)
    a = ckpt["args"]
    model = VAE(784, a["h_dim"], a["z_dim"], arch=a.get("arch", "mlp")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--z-dim", type=int, default=20)
    p.add_argument("--h-dim", type=int, default=500)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=["mlp", "conv"], default="mlp")
    p.add_argument("--iwae-k", type=int, default=1, help="importance samples for the IWAE objective; 1 = ELBO")
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE); 1 = true ELBO")
    p.add_argument("--kl-warmup", type=int, default=0, help="epochs to anneal the KL weight from 0 to beta")
    train(p.parse_args())
