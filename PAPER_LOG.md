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

---

## Completed

| Status | Paper | Authors | Year | Topic | Notes |
|--------|-------|---------|------|-------|-------|
| ✅ | Attention Is All You Need | Vaswani et al. | 2017 | Transformers | Full encoder–decoder; 41.0 test BLEU (BPE + 5-ckpt average, beam 4) on Multi30k De→En, 9.1M params, CPU; matches `nn.Transformer` to 1e-5; KV-cache, batched/pruned beam, pre-norm and large-batch comparisons (2026-09-07 → 09-18) |
| ✅ | Auto-Encoding Variational Bayes | Kingma & Welling | 2013 | VAE | MNIST log p(x) −97.8 (IW, 5k samples) at Z=20, matches paper; Z sweep 2–200 with active-units analysis; conv −90.7, conv+IWAE −90.1, β-VAE sweep (2026-09-13 → 09-15) |
| ✅ | Neural Collaborative Filtering | He et al. | 2017 | RecSys | MovieLens-1M. 8 factors reproduces: NeuMF HR@10 0.683 / NDCG 0.407 (paper 0.688 / 0.410), pre-training no gain as in paper. 64 factors: apparent 0.018 gap traced mostly to the leave-one-out split (72% held-out-item agreement; 42% of users have tied timestamps). On the authors' published split NeuMF reaches 0.696 vs paper 0.705 and the GMF>NeuMF reversal disappears. L2 ruled out; cross-split evaluation shown to leak. Adds validation-selected epochs and sparse embedding gradients (1.8x on MLP-64) (2026-09-21 → 09-23) |
