# Auto-Encoding Variational Bayes

> Kingma & Welling — 2013 — https://arxiv.org/abs/1312.6114

*Work in progress. Results and figures are filled in as training completes.*

---

## Key Idea

We want a latent-variable generative model p(x, z) = p(z) p_θ(x|z) and its posterior
p(z|x), but the posterior is intractable for any interesting decoder. The paper's answer is
to learn an approximate posterior q_φ(z|x) with a second network (the *encoder*), and train
both networks jointly by maximising a lower bound on log p(x). The key trick that makes
this work with backpropagation is the **reparameterization trick**: write z = μ + σ ⊙ ε with
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

Z = 20 for the main model (the paper sweeps Z ∈ {3, 5, 10, 20, 200}); Z = 2 for the
latent-manifold figure.

---

## Key Equations

**The evidence lower bound (ELBO), Eq. 10**

    log p(x) ≥ L(x) = E_{q(z|x)}[ log p(x|z) ] − KL( q(z|x) ‖ p(z) )

The first term rewards reconstruction; the second keeps the posterior close to the prior so
that samples from the prior decode to sensible images.

**Closed-form KL for Gaussian q and standard-normal p, Appendix B**

    KL = −½ Σ_j ( 1 + log σ_j² − μ_j² − σ_j² )

Computing this analytically instead of by sampling removes one source of gradient variance.

**Reparameterization, Section 2.4**

    z = μ(x) + σ(x) ⊙ ε,   ε ~ N(0, I)
    ∇_φ E_q[f(z)] = E_ε[ ∇_φ f(μ + σ ⊙ ε) ]

**Importance-sampled marginal likelihood, Appendix D**

    log p(x) ≈ log (1/L) Σ_i  p(x|z_i) p(z_i) / q(z_i|x),   z_i ~ q(z|x)

Tighter than the ELBO as L grows; this is what the paper reports as marginal likelihood.

---

## Files

| File | What it is |
|---|---|
| `model.py` | Encoder, decoder, reparameterization, closed-form KL, ELBO, importance-sampled log p(x), sampling and reconstruction |
| `test_model.py` | Six tests: shapes, analytic KL vs Monte Carlo, gradient flow through the reparameterization, Bernoulli likelihood, log p(x) ≥ ELBO, ELBO rises with training |
| `train.py` | MNIST training with per-epoch ELBO, KL and reconstruction tracking |

## Running it

```bash
python -m pytest test_model.py -v
python train.py --z-dim 20
```
