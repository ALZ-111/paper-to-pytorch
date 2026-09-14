"""
Tests for the VAE components. Run with:  python -m pytest test_model.py -v
"""

import math

import torch
import pytest

import model as m

B, D, H, Z = 8, 784, 64, 5


def test_1_shapes():
    vae = m.VAE(D, H, Z)
    x = torch.rand(B, D)
    logits, mu, logvar = vae(x)
    assert logits.shape == (B, D) and mu.shape == (B, Z) and logvar.shape == (B, Z)
    elbo, recon, kl = vae.elbo(x)
    assert elbo.shape == recon.shape == kl.shape == (B,)
    assert torch.allclose(elbo, recon - kl)
    assert vae.sample(3).shape == (3, D)
    assert vae.reconstruct(x).shape == (B, D)


def test_2_closed_form_kl_matches_monte_carlo():
    """The analytic KL (Appendix B) must agree with a Monte Carlo estimate of
    E_q[log q(z) - log p(z)] to within sampling error."""
    torch.manual_seed(0)
    mu = torch.randn(B, Z)
    logvar = torch.randn(B, Z) * 0.5
    analytic = m.kl_standard_normal(mu, logvar)

    n = 200_000
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(n, B, Z)
    z = mu + std * eps
    log_q = -0.5 * (eps.pow(2) + math.log(2 * math.pi) + logvar).sum(-1)
    log_p = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
    mc = (log_q - log_p).mean(0)
    assert torch.allclose(analytic, mc, atol=0.05), (analytic - mc).abs().max()
    assert (analytic >= 0).all()
    assert torch.allclose(m.kl_standard_normal(torch.zeros(1, Z), torch.zeros(1, Z)), torch.zeros(1))


def test_3_reparameterization_passes_gradients():
    """d z / d mu = 1 and d z / d sigma = eps; without the trick these would be zero."""
    mu = torch.zeros(B, Z, requires_grad=True)
    logvar = torch.zeros(B, Z, requires_grad=True)
    z = m.reparameterize(mu, logvar)
    z.sum().backward()
    assert torch.allclose(mu.grad, torch.ones(B, Z))
    assert logvar.grad.abs().sum() > 0

    # Statistics of the samples match the parameters.
    torch.manual_seed(0)
    mu = torch.tensor([[2.0]]).expand(100_000, 1)
    logvar = torch.log(torch.tensor([[0.25]])).expand(100_000, 1)
    z = m.reparameterize(mu, logvar)
    assert abs(z.mean().item() - 2.0) < 0.01
    assert abs(z.std().item() - 0.5) < 0.01


def test_4_bernoulli_log_likelihood_is_negative_bce():
    torch.manual_seed(0)
    logits = torch.randn(B, D)
    x = (torch.rand(B, D) > 0.5).float()
    p = torch.sigmoid(logits)
    manual = (x * p.log() + (1 - x) * (1 - p).log()).sum(1)
    assert torch.allclose(m.bernoulli_log_likelihood(logits, x), manual, atol=1e-4)
    assert (m.bernoulli_log_likelihood(logits, x) <= 0).all()


def test_5_importance_estimate_is_at_least_elbo():
    """log p(x) >= ELBO always (Jensen). With many importance samples the estimate
    should sit above the ELBO, and with one sample it equals the ELBO in expectation."""
    torch.manual_seed(0)
    vae = m.VAE(D, H, Z)
    x = (torch.rand(B, D) > 0.5).float()
    elbo, _, _ = vae.elbo(x, n_samples=64)
    iw = vae.log_marginal_likelihood(x, n_samples=2000, chunk=500)
    assert (iw >= elbo - 1.0).all(), (elbo - iw).max()
    assert iw.shape == (B,)


def test_6_elbo_improves_with_training():
    """A few hundred steps on a tiny synthetic dataset must raise the ELBO."""
    torch.manual_seed(0)
    vae = m.VAE(D, H, Z)
    x = (torch.rand(64, D) > 0.8).float()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    before = vae.elbo(x)[0].mean().item()
    for _ in range(200):
        loss = -vae.elbo(x)[0].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = vae.elbo(x)[0].mean().item()
    assert after > before + 50, (before, after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_7_conv_architecture():
    """Conv encoder/decoder must plug into the same VAE interface with identical shapes."""
    vae = m.VAE(D, H, Z, arch="conv")
    x = torch.rand(B, D)
    logits, mu, logvar = vae(x)
    assert logits.shape == (B, D) and mu.shape == (B, Z) and logvar.shape == (B, Z)
    elbo, recon, kl = vae.elbo(x)
    assert elbo.shape == (B,) and torch.isfinite(elbo).all()
    assert vae.sample(3).shape == (3, D) and vae.reconstruct(x).shape == (B, D)
    with pytest.raises(ValueError):
        m.VAE(D, H, Z, arch="rnn")


def test_8_iwae_bound_is_tighter_than_elbo_and_differentiable():
    """L_k >= L_1 = ELBO in expectation (Burda et al. 2015, Theorem 1), and the bound
    must carry gradients back to both encoder and decoder so it can be trained on."""
    torch.manual_seed(0)
    vae = m.VAE(D, H, Z)
    x = torch.rand(64, D)
    with torch.no_grad():
        elbo = torch.stack([vae.elbo(x)[0] for _ in range(200)]).mean()
        l1 = torch.stack([vae.iwae(x, k=1) for _ in range(200)]).mean()
        l10 = torch.stack([vae.iwae(x, k=10) for _ in range(200)]).mean()
        logpx = vae.log_marginal_likelihood(x, n_samples=2000).mean()
    assert abs(l1 - elbo) < 0.5          # k = 1 recovers the ELBO
    assert l10 > elbo                    # tighter with more samples
    assert l10 <= logpx + 0.5            # never above the (well-estimated) marginal

    bound = vae.iwae(x, k=5).mean()
    bound.backward()
    assert vae.encoder.mu.weight.grad.abs().sum() > 0
    assert vae.decoder.out.weight.grad.abs().sum() > 0
