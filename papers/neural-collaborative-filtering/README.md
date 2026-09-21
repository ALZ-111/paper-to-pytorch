# Neural Collaborative Filtering

> Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua — WWW 2017 — https://arxiv.org/abs/1708.05031

*Work in progress. Results are filled in as models are trained.*

---

## Key Idea

Matrix factorisation predicts a user's affinity for an item as the dot product of two
learned vectors. That is a fixed, linear interaction: it cannot express preferences that
depend on combinations of latent factors. NCF replaces the dot product with a learned
function. Its **GMF** branch generalises MF (an element-wise product followed by a learned
weighting), its **MLP** branch learns interactions from concatenated embeddings with a
tower of non-linear layers, and **NeuMF** fuses both. It is trained on implicit feedback
(did the user interact or not) as binary classification with sampled negatives.

---

## Setup (Section 4.1)

| | MovieLens-1M |
|---|---|
| Users / items / interactions | 6,040 / 3,706 / 1,000,209 |
| Sparsity | 95.53% |
| Feedback | implicit: every rating is a positive, the star value is dropped |
| Split | leave-one-out: each user's latest interaction is the test item |
| Test candidates | the test item ranked against 99 items the user never interacted with |
| Metrics | HR@10 (is the test item in the top 10?) and NDCG@10 (how high?) |
| Training negatives | 4 per positive, resampled each epoch, never a training item |

`data.py` builds this split from the raw ratings and caches it with fixed test negatives,
so every model is scored on the same 604,000 (user, item) pairs. The statistics match the
paper's Table 1 exactly.

---

## Files

| File | What it is |
|---|---|
| [`data.py`](data.py) | MovieLens-1M download, leave-one-out split, fixed test negatives, vectorised training-negative sampler |
| [`test_data.py`](test_data.py) | No-leakage, tie-breaking, negative-validity and determinism tests on synthetic data, plus a check of the real split against Table 1 |

## Running it

```bash
python data.py                      # download and build the split (~20 s once)
python -m pytest test_data.py -v
```
