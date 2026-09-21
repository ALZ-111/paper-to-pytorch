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
| 3 | [Neural Collaborative Filtering](papers/neural-collaborative-filtering) | 2017 | RecSys | NeuMF HR@10 0.683 / NDCG@10 0.407 on MovieLens-1M (paper 0.688 / 0.410), leakage-free pipeline |

---

## Structure
paper-to-pytorch/
├── papers/
│ └── <paper-name>/
│ ├── README.md # Summary, key ideas, results
│ ├── model.py # Core architecture
│ ├── train.py # Training loop
│ ├── utils.py # Helpers
│ └── notebook.ipynb # Walkthrough + visualizations
├── utils/ # Shared utilities across papers
└── assets/ # Diagrams, figures


---

## Setup

```bash
git clone https://github.com/ALZ-111/paper-to-pytorch.git
cd paper-to-pytorch
pip install -r requirements.txt
```

---

## About

Built by [Andrew Zhao](https://github.com/ALZ-111) — CS + Applied Math @ UC Berkeley.
Intern @ TikTok USDS working on LLM inference pipelines and recommendation systems.
