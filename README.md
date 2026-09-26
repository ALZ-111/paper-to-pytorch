# paper-to-pytorch

[![tests](https://github.com/ALZ-111/paper-to-pytorch/actions/workflows/tests.yml/badge.svg)](https://github.com/ALZ-111/paper-to-pytorch/actions/workflows/tests.yml)

PyTorch implementations of ML research papers from scratch.

Each folder in `papers/` contains a clean, well-commented implementation of a landmark paper — built to understand the ideas deeply, not just run the code.

---

## Papers Implemented

| # | Paper | Year | Topic | Result |
|---|-------|------|-------|--------|
| 1 | [Attention Is All You Need](papers/attention-is-all-you-need) | 2017 | Transformers | 41.0 BLEU on Multi30k De→En from scratch (own BPE + checkpoint averaging), verified against `nn.Transformer`; KV-cache and batched beam search |
| 2 | [Auto-Encoding Variational Bayes](papers/auto-encoding-variational-bayes) | 2013 | VAE | log p(x) −97.8 on MNIST matches paper; conv+IWAE variant −90.1; β-VAE and latent-dim sweeps with active-units analysis |
| 3 | [Neural Collaborative Filtering](papers/neural-collaborative-filtering) | 2017 | RecSys | NeuMF HR@10 0.683 vs paper 0.688 at 8 factors; at 64, most of the apparent gap traced to a train/test split disagreement (0.696 vs 0.705 on the authors' own split) |

---

## Layout

```
paper-to-pytorch/
├── papers/<paper-name>/
│   ├── README.md        # the paper's ideas, equations, results against the published numbers
│   ├── model.py         # the architecture, commented against the paper's sections
│   ├── train.py         # training with the paper's recipe
│   ├── test_*.py        # correctness tests: shapes, invariants, leakage, chance levels
│   ├── visualize.py     # every figure in that README
│   └── results.json     # per-epoch metrics for each run reported
├── utils/               # shared across papers: device, seeding, checkpoints, results, plotting
├── assets/<paper>/      # figures
└── Makefile             # make test, make figures
```

Each paper folder is self-contained and run from inside itself (`cd papers/... && python
train.py`); a two-line `_bootstrap.py` puts the repo root on `sys.path` so `utils` imports
work. Checkpoints and datasets are git-ignored, so a fresh clone downloads and retrains.

---

## Running it

```bash
pip install -r requirements.txt
make test        # all 63 tests across the three papers and utils
make figures     # regenerate the figures that need no trained checkpoint
```

Each paper's README has its own training commands and how long they take on a CPU.

---

## About

Built by [Andrew Zhao](https://github.com/ALZ-111) — CS + Applied Math @ UC Berkeley.
Intern @ TikTok USDS working on LLM inference pipelines and recommendation systems.
