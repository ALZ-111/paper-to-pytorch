# Auto-Encoding Variational Bayes

> Kingma & Welling — 2013 — https://arxiv.org/abs/1312.6114

A from-scratch PyTorch variational autoencoder on MNIST, reproducing the paper's
lower-bound and marginal-likelihood numbers, its latent-dimension sweep, and its
2-D manifold figure. Every run trains in under four minutes on a laptop CPU.

<p align="center">
  <img src="../../assets/vae/latent_manifold.png" width="55%">
  <br><em>A 2-dimensional latent space, decoded on a 20×20 grid of prior quantiles (paper Figure 4).
  Digits morph continuously into one another: 1 → 7 → 9 → 4 across the top, 2 → 6 → 0 along the bottom.</em>
</p>

---

## Results

### MNIST test set

| | Z = 20, this implementation | Paper (Z = 20, Fig. 2–3) |
|---|---|---|
| **log p(x), importance-weighted** | **−97.8** | ≈ −97 to −100 |
| ELBO (16-sample) | −103.1 | ≈ −100 |
| Reconstruction / KL split of the ELBO | −76.3 / 26.8 | |
| Parameters | 0.82M | same architecture (500 tanh units) |
| Training | 50 epochs, 3.4 min, 4-core CPU | Adagrad, ~10⁸ samples |

Headline numbers use all 10,000 test images and 5,000 importance samples per image from
q(z|x), the paper's own protocol (Appendix D); the sweep table below uses 2,000 images and
1,000 samples, which agrees to 0.1 nat. ELBO uses 16 posterior samples. Pixels are treated as Bernoulli
with dynamic binarisation during training.

### Latent-dimension sweep (paper Figure 2)

| Z | test ELBO | log p(x) | gap |
|---|---|---|---|
| 2 | −146.3 | −142.9 | 3.4 |
| 3 | −136.1 | −132.5 | 3.6 |
| 5 | −121.1 | −117.6 | 3.6 |
| 10 | −106.5 | −102.6 | 4.0 |
| 20 | −102.6 | **−97.7** | 4.9 |
| 200 | −105.6 | −99.4 | 6.2 |

Two of the paper's observations reproduce: the bound and the likelihood improve with Z up
to about 20, and a large latent space (Z = 200) does **not** overfit badly, because the KL
term switches off dimensions the decoder does not need (the paper's "more latent variables
does not result in more overfitting"). The ELBO-to-log p(x) gap widens with Z, meaning the
diagonal-Gaussian posterior becomes a looser fit as the latent space grows.

<p align="center"><img src="../../assets/vae/training_curves.png" width="100%"></p>

### Samples and reconstructions

<p align="center">
  <img src="../../assets/vae/samples.png" width="42%">
  <img src="../../assets/vae/latent_scatter.png" width="50%">
  <br><em>Left: 100 decodes of z ~ N(0, I) from the Z = 20 model. Right: posterior means of 10k test
  digits in the Z = 2 model, coloured by label the model never saw.</em>
</p>

<p align="center"><img src="../../assets/vae/reconstructions.png" width="100%"></p>

---

## Key Idea

We want a latent-variable generative model p(x, z) = p(z) p_θ(x|z) and its posterior
p(z|x), but the posterior is intractable for any interesting decoder. The paper's answer is
to learn an approximate posterior q_φ(z|x) with a second network (the *encoder*), and train
both networks jointly by maximising a lower bound on log p(x). The trick that makes this
work with backpropagation is the **reparameterization trick**: write z = μ + σ ⊙ ε with
ε ~ N(0, I), so the sample is a deterministic, differentiable function of the encoder's
outputs and the gradient can flow through the sampling step.

---

## Architecture (Appendix C of the paper)

```
x (784) ─► Linear 500, tanh ─► μ (Z), log σ² (Z)        encoder q_φ(z|x)
                                     │
                            z = μ + σ ⊙ ε,  ε ~ N(0, I)  reparameterization
                                     │
z (Z) ─► Linear 500, tanh ─► Bernoulli logits (784)     decoder p_θ(x|z)
```

---

## Key Equations

**The evidence lower bound (ELBO), Eq. 10**

    log p(x) ≥ L(x) = E_{q(z|x)}[ log p(x|z) ] − KL( q(z|x) ‖ p(z) )

The first term rewards reconstruction; the second keeps the posterior close to the prior so
that samples from the prior decode to sensible images.

**Closed-form KL for Gaussian q and standard-normal p, Appendix B**

    KL = −½ Σ_j ( 1 + log σ_j² − μ_j² − σ_j² )

Computing this analytically instead of by sampling removes one source of gradient variance.
`test_model.py` checks it against a 200,000-sample Monte Carlo estimate.

**Reparameterization, Section 2.4**

    z = μ(x) + σ(x) ⊙ ε,   ε ~ N(0, I)
    ∇_φ E_q[f(z)] = E_ε[ ∇_φ f(μ + σ ⊙ ε) ]

**Importance-sampled marginal likelihood, Appendix D**

    log p(x) ≈ log (1/L) Σ_i  p(x|z_i) p(z_i) / q(z_i|x),   z_i ~ q(z|x)

Always ≥ the ELBO in expectation and tighter as L grows; computed with `logsumexp`.

---

## Files

| File | What it is |
|---|---|
| [`model.py`](model.py) | Encoder, decoder, reparameterization, closed-form KL, ELBO, importance-sampled log p(x), sampling and reconstruction, all commented against the paper's sections |
| [`test_model.py`](test_model.py) | Six tests: shapes; analytic KL vs Monte Carlo; gradient flow and sample statistics through the reparameterization; Bernoulli likelihood; log p(x) ≥ ELBO; ELBO rises with training |
| [`train.py`](train.py) | MNIST training with per-epoch ELBO, reconstruction and KL |
| [`evaluate.py`](evaluate.py) | Importance-weighted log p(x) for every trained latent size |
| [`visualize.py`](visualize.py) | All figures above |
| [`results.json`](results.json) | Per-epoch history and evaluation for all six runs |

## Running it

```bash
python -m pytest test_model.py -v                 # 6 tests, ~5 s
python train.py --z-dim 20 --epochs 50            # 3.4 min on CPU
python train.py --z-dim 2  --epochs 50            # for the manifold figure
for z in 3 5 10 200; do python train.py --z-dim $z --epochs 30; done
python evaluate.py --n-test 2000 --samples 1000   # ~2 min
python visualize.py
```

---

## What I Learned

- **The KL term is a learned bottleneck.** With Z = 200 the KL settles at about 30 nats,
  barely above Z = 20's 27: the encoder drives the variance of unused dimensions to 1 and
  their mean to 0, paying zero KL for them. This is why the model does not overfit as Z grows.
- **Analytic KL matters.** Estimating the KL by sampling adds variance to every gradient
  step; the closed form makes the one-sample ELBO estimator good enough to train on, which is
  the paper's Section 2.4 point about the SGVB estimator.
- **Dynamic binarisation is quiet data augmentation.** Resampling x ~ Bernoulli(intensity)
  each minibatch keeps train and test ELBO within 5 nats after 50 epochs.
- **The manifold grid needs the inverse CDF.** Spacing the 2-D grid linearly wastes most
  tiles on the prior's tails; mapping uniform quantiles through Φ⁻¹ covers the mass evenly
  and gives the paper's figure.
- **Logits everywhere.** Computing the Bernoulli likelihood from logits with
  `binary_cross_entropy_with_logits` avoids log(0) when the decoder becomes confident.

## Resume-ready summary

> Implemented a variational autoencoder from scratch in PyTorch (Kingma & Welling, 2013),
> including the reparameterization trick, closed-form Gaussian KL and an importance-sampled
> marginal-likelihood estimator; reproduced the paper's MNIST results (log p(x) = −97.8 nats
> at Z = 20) and its latent-dimension sweep in under 20 minutes of laptop-CPU training, with
> unit tests verifying the analytic KL against Monte Carlo to 0.05 nats.
