# Paper Log

Tracking papers I've read, am implementing, or plan to implement.

## Status Key
- 📋 `queued` — on the list
- 🔨 `in progress` — currently implementing
- ✅ `done` — implemented and documented

---

## Queue

| Status | Paper | Authors | Year | Topic |
|--------|-------|---------|------|-------|
| 📋 | Deep Residual Learning for Image Recognition | He et al. | 2015 | ResNet |
| 📋 | Generative Adversarial Networks | Goodfellow et al. | 2014 | GAN |
| 📋 | BERT | Devlin et al. | 2018 | NLP |
| 📋 | Neural Collaborative Filtering | He et al. | 2017 | RecSys |

---

## Completed

| Status | Paper | Authors | Year | Topic | Notes |
|--------|-------|---------|------|-------|-------|
| ✅ | Attention Is All You Need | Vaswani et al. | 2017 | Transformers | Full encoder–decoder; 40.7 test BLEU (beam 4) on Multi30k De→En, 9.1M params, 72 min CPU; matches `nn.Transformer` to 1e-5 (2026-09-07 → 09-09) |
| ✅ | Auto-Encoding Variational Bayes | Kingma & Welling | 2013 | VAE | MNIST log p(x) −97.8 (IW, 5k samples) at Z=20, matches paper; Z sweep 2–200 with active-units analysis; conv variant −90.7, IWAE k=5 −96.9 (2026-09-13 → 09-14) |
