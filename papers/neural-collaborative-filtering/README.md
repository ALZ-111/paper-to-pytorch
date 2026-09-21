# Neural Collaborative Filtering

> Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua — WWW 2017 — https://arxiv.org/abs/1708.05031

A from-scratch PyTorch reproduction of NCF on MovieLens-1M. NeuMF lands within
0.005 HR@10 of the paper at 8 predictive factors, and the paper's finding that
pre-training does not help at this size reproduces too. Every model trains in under
20 minutes on a laptop CPU.

<p align="center"><img src="../../assets/ncf/training_curves.png" width="100%"></p>

---

## Results

### MovieLens-1M, 8 predictive factors, leave-one-out, 1 positive vs 99 negatives

| Model | Params | HR@10 | NDCG@10 | Paper HR@10 / NDCG@10 | Train time |
|---|---|---|---|---|---|
| GMF | 78k | 0.645 | 0.366 | | 8.7 min |
| MLP (3 hidden layers) | 315k | 0.679 | 0.399 | 0.671 / – (Table 3) | 15.8 min |
| **NeuMF, from scratch** | 393k | **0.683** | **0.407** | 0.688 / 0.410 (Table 2) | 17.8 min |
| NeuMF, pre-trained | 393k | 0.675 | 0.395 | 0.684 / 0.403 (Table 2) | 11.5 min |
| Random ranking | | 0.100 | 0.046 | | |

Scores are from the final (20th) epoch. The authors' code instead reports the best epoch
by test HR, which chooses the model with test labels; by that protocol the pre-trained
NeuMF scores 0.681 / 0.399 (epoch 1) and the others are unchanged, because their best
epoch was their last. The untrained models all score HR ≈ 0.10, exactly chance for
ranking 1 item among 100, which checks the evaluation end to end.

- **NeuMF beats both of its branches**, as the paper claims: +0.004 HR over MLP and
  +0.038 over GMF, with the larger margin on NDCG (+0.008 over MLP), meaning it ranks the
  held-out item higher, not just into the top 10 more often.
- **Pre-training does not help at 8 factors, in the paper or here.** The pre-trained
  NeuMF starts as the 0.5/0.5 blend of GMF and MLP (HR 0.677 before any training), gains
  0.004 in the first SGD epoch, then drifts down. The paper's Table 2 shows the same at 8
  factors (0.684 with vs 0.688 without); its pre-training gains appear at 32 and 64.
- **The ensemble is worse than its better half.** Averaging GMF (0.645) and MLP (0.679)
  logits gives 0.677: a weaker model dilutes a stronger one. Joint training is what makes
  the fusion pay off, and it is also why pre-training has little left to add.

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
| [`model.py`](model.py) | GMF, MLP and NeuMF (Eqs. 9–12), sized as in the authors' code, with NeuMF pre-training from trained GMF and MLP |
| [`metrics.py`](metrics.py) | HR@10 and NDCG@10 over all 604k test pairs in a few batched passes |
| [`visualize.py`](visualize.py) | Training-curve figure |
| [`train.py`](train.py) | Training with the paper's recipe, per-epoch test HR/NDCG, final-epoch and best-epoch reporting, NeuMF pre-training |
| [`test_model.py`](test_model.py) | GMF reduces to MF, tower shapes, no dead parameters, pre-trained NeuMF equals the α-blend of its parents, metrics on hand-computed cases, chance and oracle scores |

## Running it

```bash
python data.py                      # download and build the split (~20 s once)
python -m pytest -v                 # 15 tests
python train.py --model gmf --factors 8     # ~9 min on CPU
python train.py --model mlp --factors 8     # ~16 min
python train.py --model neumf --factors 8
python train.py --model neumf --factors 8 --pretrain   # needs the GMF and MLP above
```

---

## What I Learned

- **Leave-one-out evaluation hides a model-selection trap.** Scoring the test set every
  epoch and reporting the best one is the reference implementation's protocol, and it
  quietly selects on test labels. Reporting the final epoch costs nothing here (best =
  final for three of four models) and removes the question.
- **Where you reject negatives from is a leakage decision.** Training negatives are
  rejected against the user's *training* items only. Also rejecting the held-out item
  would tell the model, indirectly, which item is the answer.
- **Chance level is the best end-to-end test.** An untrained model must score HR@10 ≈
  0.10 on 1-in-100 ranking; a degenerate one that ties everything must score 0, which is
  why ties count against the positive. Both are unit tests.
- **The paper's prose and code disagree on the MLP tower** (32 → 16 → 8 vs
  64 → 32 → 16 → 8). The code's version matches the "three hidden layers" of Table 3 and
  reproduces its 0.671, so that is what is implemented.
- **Vectorise the sampler, not just the model.** Rejection-sampling 4 million negatives
  per epoch with encoded keys and a binary search takes 1.3 s; the reference per-sample
  loop would dominate training time.

## Next

The paper's headline numbers (HR 0.730 with pre-training) are at 64 predictive factors,
where pre-training finally helps. That sweep (factors 16, 32, 64) is the natural next
step: `python train.py --model {gmf,mlp,neumf} --factors 64`, then `--pretrain`.

## Resume-ready summary

> Reproduced Neural Collaborative Filtering (He et al., 2017) from scratch in PyTorch on
> MovieLens-1M: GMF, MLP and NeuMF with pre-training, a leakage-free leave-one-out
> pipeline and a vectorised negative sampler (4M negatives/epoch in 1.3 s). NeuMF reaches
> HR@10 0.683 / NDCG@10 0.407, within 0.005 of the paper, and reproduces its finding that
> pre-training only helps at larger factor sizes; 15 unit tests including leakage,
> tie-handling and chance-level checks.
