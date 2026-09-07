# Attention Is All You Need

> Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin — 2017 — https://arxiv.org/abs/1706.03762

---

## Key Idea

Sequence models before this paper (RNNs, LSTMs) processed tokens one at a time, so they
were slow to train and struggled to relate tokens that were far apart. The Transformer drops
recurrence entirely and lets every token look at every other token directly through
*attention*. Because attention is just matrix multiplications, the whole sequence is processed
in parallel, and the path between any two positions is length 1.

---

## Architecture

Encoder–decoder, both stacks of N = 6 identical layers.

```
Encoder layer:                  Decoder layer:
  x ─► Multi-Head Self-Attn       y ─► Masked Multi-Head Self-Attn
    ─► Add & LayerNorm              ─► Add & LayerNorm
    ─► Feed-Forward                 ─► Multi-Head Cross-Attn (Q from decoder, K/V from encoder)
    ─► Add & LayerNorm              ─► Add & LayerNorm
                                    ─► Feed-Forward
                                    ─► Add & LayerNorm
```

Base model: d_model = 512, h = 8 heads, d_k = d_v = 64, d_ff = 2048, dropout = 0.1.

---

## Key Equations

**Scaled dot-product attention**

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Divide by √d_k so the dot products don't grow with dimension and push softmax into
regions with tiny gradients.

**Multi-head attention**

    head_i = Attention(Q W_iQ, K W_iK, V W_iV)
    MultiHead(Q, K, V) = Concat(head_1, …, head_h) W_O

Each head gets its own low-dimensional projection, so heads can specialise.

**Position-wise feed-forward**

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

**Sinusoidal positional encoding**

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Attention is permutation-invariant, so position has to be injected explicitly.

**Residual + LayerNorm** (post-norm in the original paper)

    LayerNorm(x + Sublayer(x))

---

## Build Order

Each piece has a shape test in `test_model.py`; implement top to bottom and run the tests.

1. `scaled_dot_product_attention` — the core function, with optional mask
2. `MultiHeadAttention` — projections, head split/merge
3. `PositionwiseFeedForward`
4. `PositionalEncoding`
5. `EncoderLayer`, `DecoderLayer`
6. `Encoder`, `Decoder`, `Transformer` — plus `make_causal_mask` and `make_pad_mask`

---

## My Implementation

- **Model:** `model.py` — every component from the paper, heavily commented, plus `greedy_decode`
- **Tests:** `test_model.py` — shape tests per component and a causality check on the full model
- **Training:** `train.py` — Noam LR schedule, label smoothing, Adam(β₂=0.98) as in Section 5.3

```bash
python -m pytest test_model.py -v   # 7 tests
python train.py                     # toy task, ~40s on CPU
```

---

## Results

The paper trains on WMT14 En–De for 12 hours on 8 P100s; not reproduced here. Instead the
model is verified end to end on a toy sequence-reversal task: encoder sees `[3, 9, 4, 1]`,
decoder must emit `[1, 4, 9, 3]`. Solving it requires the decoder to cross-attend to
position S−1−i of the source.

| Metric | Paper (WMT14 En–De, base) | Mine (reverse task, 2 layers, d=128) |
|--------|---------------------------|--------------------------------------|
| Parameters | 65M | 0.93M |
| Exact-match seq accuracy | – | 99.2% after 600 steps |
| BLEU | 27.3 | – |

Training loss plateaus around 0.65 rather than 0; that is the floor imposed by label
smoothing 0.1, not underfitting.

---

## What I Learned

*Fill in as you go.* Some things worth noticing while reading the code:

- The multi-head "Concat" is a `view` + `transpose`; there is no real concatenation.
- Masks combine by broadcasting: pad mask `(B,1,1,T)` & causal mask `(1,1,T,T)` → `(B,1,T,T)`.
- The decoder input and loss target are the same sequence shifted by one (teacher forcing).
- Embeddings are multiplied by √d_model so the positional encoding doesn't dominate them.
