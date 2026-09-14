"""
Auto-Encoding Variational Bayes - Kingma & Welling, 2013
Paper: https://arxiv.org/abs/1312.6114

Variational autoencoder (VAE) for binarised MNIST, following the paper's Appendix C:
an MLP encoder q_phi(z|x) = N(mu(x), diag(sigma(x)^2)), a standard-normal prior p(z),
and an MLP Bernoulli decoder p_theta(x|z).

Shape conventions: B = batch, D = data dim (784), H = hidden width, Z = latent dim.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Recognition model q_phi(z | x): maps x to the mean and log-variance of a
    diagonal Gaussian over z. Outputting log sigma^2 rather than sigma keeps the
    variance positive without a constraint and is numerically friendlier."""

    def __init__(self, x_dim=784, h_dim=500, z_dim=20):
        super().__init__()
        self.fc = nn.Linear(x_dim, h_dim)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        """x: (B, D) in [0, 1] -> mu (B, Z), logvar (B, Z)"""
        h = torch.tanh(self.fc(x))  # the paper uses tanh hidden units
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """Generative model p_theta(x | z): maps z to Bernoulli logits over the pixels."""

    def __init__(self, x_dim=784, h_dim=500, z_dim=20):
        super().__init__()
        self.fc = nn.Linear(z_dim, h_dim)
        self.out = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        """z: (B, Z) -> logits (B, D). Apply sigmoid to get pixel probabilities."""
        return self.out(torch.tanh(self.fc(z)))


class ConvEncoder(nn.Module):
    """Convolutional recognition model. Not in the 2013 paper (which predates the
    conv-VAE literature) but the natural upgrade for image data: two stride-2 convs
    reduce 28x28 to 7x7 before a small MLP head produces mu and log sigma^2."""

    def __init__(self, z_dim=20, h_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),   # 28 -> 14
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 14 -> 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, h_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        """x: (B, 784) -> mu (B, Z), logvar (B, Z)"""
        h = self.conv(x.view(-1, 1, 28, 28))
        return self.mu(h), self.logvar(h)


class ConvDecoder(nn.Module):
    """Mirror of ConvEncoder using transposed convolutions; outputs Bernoulli logits."""

    def __init__(self, z_dim=20, h_dim=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(),
                                nn.Linear(h_dim, 64 * 7 * 7), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),              # 14 -> 28
        )

    def forward(self, z):
        """z: (B, Z) -> logits (B, 784)"""
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h).view(-1, 784)


def reparameterize(mu, logvar):
    """The reparameterization trick (Section 2.4).

    We need gradients of E_{z ~ q(z|x)}[f(z)] with respect to the parameters of q.
    Sampling z directly blocks the gradient, so instead sample noise eps ~ N(0, I)
    and set z = mu + sigma * eps. Now z is a deterministic, differentiable function
    of (mu, sigma) and the randomness lives in eps, which needs no gradient.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_standard_normal(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, I) ) per example, in closed form (Appendix B):

        -KL = 1/2 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)

    Returns (B,). Having this analytically, rather than by sampling, removes one
    source of gradient variance; only the reconstruction term is estimated by sampling.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def bernoulli_log_likelihood(logits, x):
    """log p(x | z) for a factorised Bernoulli decoder, per example: (B,).
    Equivalent to minus the binary cross-entropy summed over pixels. Uses the logits
    form for numerical stability."""
    return -F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)


class VAE(nn.Module):
    def __init__(self, x_dim=784, h_dim=500, z_dim=20, arch="mlp"):
        super().__init__()
        self.z_dim = z_dim
        self.arch = arch
        if arch == "mlp":
            self.encoder = Encoder(x_dim, h_dim, z_dim)
            self.decoder = Decoder(x_dim, h_dim, z_dim)
        elif arch == "conv":
            assert x_dim == 784, "conv architecture assumes 28x28 inputs"
            self.encoder = ConvEncoder(z_dim)
            self.decoder = ConvDecoder(z_dim)
        else:
            raise ValueError(f"unknown arch {arch!r}")

    def forward(self, x):
        """x: (B, D) -> logits (B, D), mu (B, Z), logvar (B, Z)"""
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def elbo(self, x, n_samples=1):
        """Per-example evidence lower bound (Equation 10 of the paper):

            L(x) = E_q[log p(x|z)] - KL(q(z|x) || p(z))

        The expectation is estimated with n_samples reparameterised draws (the paper
        uses 1 per datapoint when the minibatch is large enough). Returns
        (elbo, recon_term, kl_term), each of shape (B,). Training maximises elbo.mean().
        """
        mu, logvar = self.encoder(x)
        kl = kl_standard_normal(mu, logvar)
        recon = 0.0
        for _ in range(n_samples):
            z = reparameterize(mu, logvar)
            recon = recon + bernoulli_log_likelihood(self.decoder(z), x)
        recon = recon / n_samples
        return recon - kl, recon, kl

    def iwae(self, x, k=5):
        """Importance-weighted bound of Burda, Grosse & Salakhutdinov (2015), per example:

            L_k(x) = E_{z_1..z_k ~ q}[ log (1/k) sum_i  p(x|z_i) p(z_i) / q(z_i|x) ]

        L_1 is the ELBO; L_k is tighter as k grows and approaches log p(x). Unlike
        log_marginal_likelihood this keeps the graph, so it can be trained on: the
        gradient reweights samples by their normalised importance weights, which lets
        the encoder learn a broader posterior than the ELBO permits. Returns (B,).
        """
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(k, *mu.shape, device=x.device)               # (k, B, Z)
        z = mu + std * eps
        log_p_x_z = bernoulli_log_likelihood(
            self.decoder(z.view(-1, self.z_dim)), x.repeat(k, 1)).view(k, -1)  # (k, B)
        log_p_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        log_q_z = -0.5 * (eps.pow(2) + math.log(2 * math.pi) + logvar).sum(-1)
        log_w = log_p_x_z + log_p_z - log_q_z                          # (k, B)
        return torch.logsumexp(log_w, dim=0) - math.log(k)

    @torch.no_grad()
    def log_marginal_likelihood(self, x, n_samples=5000, chunk=500):
        """Importance-sampled estimate of log p(x) (Section 5 / Appendix D of the paper):

            log p(x) ~= log mean_i [ p(x|z_i) p(z_i) / q(z_i|x) ],   z_i ~ q(z|x)

        computed in log space with logsumexp. With enough samples this is a much
        tighter estimate than the ELBO and is what the paper reports as
        "marginal likelihood". Returns (B,).
        """
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        log_w = []
        for _ in range(0, n_samples, chunk):
            k = min(chunk, n_samples - len(log_w) * chunk) if log_w else min(chunk, n_samples)
            eps = torch.randn(k, *mu.shape, device=x.device)          # (k, B, Z)
            z = mu + std * eps
            log_p_x_z = bernoulli_log_likelihood(
                self.decoder(z.view(-1, self.z_dim)), x.repeat(k, 1)).view(k, -1)  # (k, B)
            log_p_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
            log_q_z = -0.5 * (eps.pow(2) + math.log(2 * math.pi) + logvar).sum(-1)
            log_w.append(log_p_x_z + log_p_z - log_q_z)
        log_w = torch.cat(log_w, dim=0)                                 # (n_samples, B)
        return torch.logsumexp(log_w, dim=0) - math.log(log_w.size(0))

    @torch.no_grad()
    def sample(self, n, device=None):
        """Draw n images from the prior: z ~ N(0, I), x ~ p(x|z). Returns probabilities (n, D)."""
        z = torch.randn(n, self.z_dim, device=device)
        return torch.sigmoid(self.decoder(z))

    @torch.no_grad()
    def reconstruct(self, x):
        """Posterior mean reconstruction: decode mu(x). Returns probabilities (B, D)."""
        mu, _ = self.encoder(x)
        return torch.sigmoid(self.decoder(mu))


if __name__ == "__main__":
    model = VAE()
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
