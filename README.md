# paper-to-pytorch

[![tests](https://github.com/ALZ-111/paper-to-pytorch/actions/workflows/tests.yml/badge.svg)](https://github.com/ALZ-111/paper-to-pytorch/actions/workflows/tests.yml)

PyTorch implementations of ML research papers from scratch.

Each folder in `papers/` contains a clean, well-commented implementation of a landmark paper — built to understand the ideas deeply, not just run the code.

---

## Papers Implemented

| # | Paper | Year | Topic | Result |
|---|-------|------|-------|--------|
| 1 | [Attention Is All You Need](papers/attention-is-all-you-need) | 2017 | Transformers | 40.7 BLEU on Multi30k De→En from scratch, verified against `nn.Transformer` |

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
