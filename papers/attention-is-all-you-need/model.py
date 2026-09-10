"""
Attention Is All You Need - Vaswani et al., 2017
Paper: https://arxiv.org/abs/1706.03762

Encoder-decoder Transformer in PyTorch, built from scratch.

Shape conventions used throughout:
    B = batch size, S = source length, T = target length,
    D = d_model, H = number of heads, Dk = D // H

Defaults to the original "post-norm" layout, LayerNorm(x + Sublayer(x)).
Pass pre_norm=True for the modern variant, x + Sublayer(LayerNorm(x)) (GPT-2 onward),
which trains more stably at depth; see SublayerConnection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Scaled dot-product attention  (Section 3.2.1)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    q: (B, H, T, Dk)   k: (B, H, S, Dk)   v: (B, H, S, Dv)
    mask: broadcastable to (B, H, T, S); True/1 = attend, False/0 = block.

    Returns:
        out:  (B, H, T, Dv)
        attn: (B, H, T, S)  the softmax weights (useful for visualisation)
    """
    d_k = q.size(-1)

    # Similarity of every query with every key: (B, H, T, Dk) @ (B, H, Dk, S) -> (B, H, T, S).
    # Each row i holds the raw "how much should query i look at key j" scores.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    # Why sqrt(d_k)? If q and k entries have unit variance, their dot product has
    # variance d_k. Large scores make softmax nearly one-hot, and its gradient
    # vanishes. Scaling restores unit variance regardless of head size.

    if mask is not None:
        # Set blocked positions to a huge negative number so softmax gives them ~0.
        # (Using -inf works too but produces NaN if a whole row is masked.)
        scores = scores.masked_fill(mask == 0, -1e9)

    # Normalise over the key axis so each query's weights sum to 1.
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    # Weighted sum of values: (B, H, T, S) @ (B, H, S, Dv) -> (B, H, T, Dv).
    out = attn @ v
    return out, attn


# ---------------------------------------------------------------------------
# 2. Multi-head attention  (Section 3.2.2)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    MultiHead(Q, K, V) = Concat(head_1..head_h) W^O

    Rather than h separate (D x Dk) projection matrices per input, we use one
    (D x D) linear layer and reshape its output into h chunks of Dk. That is
    mathematically identical and much faster.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None  # last attention weights, kept for inspection

    def _split_heads(self, x):
        """(B, L, D) -> (B, H, L, Dk): carve D into H heads, put H before L so
        the attention matmul batches over (B, H)."""
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        """(B, H, L, Dk) -> (B, L, D): inverse of _split_heads (the Concat)."""
        B, _, L, _ = x.shape
        # transpose makes the tensor non-contiguous, so .contiguous() before .view
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None, cache=None, static_kv=False):
        """
        query: (B, T, D)   key/value: (B, S, D)   mask: broadcastable to (B, H, T, S)
        Returns (B, T, D).

        Self-attention:  query = key = value = x
        Cross-attention: query = decoder state, key = value = encoder output

        cache: optional dict used during incremental decoding. Attention is the only
        place where a new token interacts with earlier ones, so it is the only place
        that needs to remember anything between steps:
          - static_kv=False (decoder self-attention): the new token's K/V rows are
            appended to cache["k"], cache["v"], and the query attends over all of them.
            Each step then costs O(t) instead of re-running the whole prefix, O(t^2).
          - static_kv=True (cross-attention): K/V of the encoder output never change,
            so they are projected once and reused for every step.
        """
        q = self._split_heads(self.w_q(query))  # (B, H, T, Dk)
        if cache is not None and static_kv and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            k = self._split_heads(self.w_k(key))    # (B, H, S, Dk)
            v = self._split_heads(self.w_v(value))  # (B, H, S, Dk)
            if cache is not None:
                if not static_kv and "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v

        out, self.attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        return self.w_o(self._merge_heads(out))  # (B, T, D)


# ---------------------------------------------------------------------------
# 3. Position-wise feed-forward  (Section 3.3)
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    "Position-wise" means the same two-layer MLP is applied to every token
    independently; there is no mixing across the sequence here. Attention mixes
    across positions, the FFN transforms each position. d_ff is usually 4 * d_model.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ---------------------------------------------------------------------------
# 4. Positional encoding  (Section 3.5)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Attention has no notion of order: shuffle the inputs and you shuffle the
    outputs identically. So we add a fixed, position-dependent vector to each
    token embedding. Sinusoids of geometrically spaced wavelengths (2*pi up to
    10000*2*pi) let the model attend by relative offset, since PE(pos + k) is a
    linear function of PE(pos) for any fixed k.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                             # (max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 1 / 10000^(2i/D), computed in log space for numerical stability.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                              # (D/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        # register_buffer: saved with the model and moved by .to(device),
        # but not a trainable parameter.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x, offset=0):
        """x: (B, T, D) token embeddings -> (B, T, D) with positions added.
        offset: index of the first position in x (non-zero during incremental decoding)."""
        x = x + self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 5. Encoder / decoder layers  (Section 3.1)
# ---------------------------------------------------------------------------
class SublayerConnection(nn.Module):
    """Residual connection followed by LayerNorm: LayerNorm(x + Dropout(sublayer(x))).

    The residual path lets gradients flow straight through N layers, and LayerNorm
    keeps activations at a stable scale. Taking `sublayer` as a callable lets the
    same wrapper serve attention and feed-forward blocks.
    """

    def __init__(self, d_model, dropout=0.1, pre_norm=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, sublayer):
        if self.pre_norm:
            # Pre-norm (Xiong et al., 2020; used by GPT-2 onward): normalise the input
            # to the sublayer and leave the residual stream un-normalised. Gradients
            # then flow through a clean identity path, so deep stacks train without
            # careful warmup. Requires a final LayerNorm at the top of the stack.
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """Self-attention -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_norm=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sub1 = SublayerConnection(d_model, dropout, pre_norm)
        self.sub2 = SublayerConnection(d_model, dropout, pre_norm)

    def forward(self, x, src_mask=None):
        """x: (B, S, D) -> (B, S, D)"""
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, src_mask))
        return self.sub2(x, self.ff)


class DecoderLayer(nn.Module):
    """Masked self-attention -> Add&Norm -> cross-attention -> Add&Norm -> FFN -> Add&Norm.

    Cross-attention is where the decoder reads the source sentence: queries come
    from the decoder, keys and values from the encoder output ("memory").
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_norm=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sub1 = SublayerConnection(d_model, dropout, pre_norm)
        self.sub2 = SublayerConnection(d_model, dropout, pre_norm)
        self.sub3 = SublayerConnection(d_model, dropout, pre_norm)

    def forward(self, x, memory, tgt_mask=None, src_mask=None, cache=None):
        """x: (B, T, D)  memory: (B, S, D) encoder output -> (B, T, D)
        cache: {"self": {}, "cross": {}} for incremental decoding, else None."""
        self_c = cache["self"] if cache is not None else None
        cross_c = cache["cross"] if cache is not None else None
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, tgt_mask, cache=self_c))
        x = self.sub2(x, lambda x: self.cross_attn(x, memory, memory, src_mask,
                                                   cache=cross_c, static_kv=True))
        return self.sub3(x, self.ff)


# ---------------------------------------------------------------------------
# 6. Masks, full stacks and the Transformer
# ---------------------------------------------------------------------------
def make_pad_mask(tokens, pad_idx):
    """tokens: (B, L) -> (B, 1, 1, L) bool, True where token is NOT padding.

    The two singleton dims broadcast over heads and query positions: every query
    is blocked from attending to padding keys.
    """
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size, device=None):
    """-> (1, 1, size, size) bool lower-triangular mask.

    Position i may attend to positions <= i only. Without this the decoder could
    read the token it is supposed to predict during teacher-forced training.
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, pre_norm=False):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout, pre_norm) for _ in range(n_layers)]
        )
        # Pre-norm leaves the residual stream un-normalised, so normalise once at the end.
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(self, src, src_mask=None):
        """src: (B, S) token ids -> (B, S, D)"""
        # Section 3.4: embeddings are scaled by sqrt(d_model) so they are not
        # drowned out by the positional encoding (which has entries in [-1, 1]).
        x = self.pos(self.embed(src) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.final_norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, pre_norm=False):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout, pre_norm) for _ in range(n_layers)]
        )
        # Pre-norm leaves the residual stream un-normalised, so normalise once at the end.
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None, caches=None, offset=0):
        """tgt: (B, T) token ids, memory: (B, S, D) -> (B, T, D)
        caches: one {"self", "cross"} dict per layer for incremental decoding, in which
        case tgt holds only the newest token(s) and offset is their first position."""
        x = self.pos(self.embed(tgt) * math.sqrt(self.d_model), offset=offset)
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, tgt_mask, src_mask, cache=caches[i] if caches else None)
        return self.final_norm(x)

    def new_caches(self):
        return [{"self": {}, "cross": {}} for _ in self.layers]


class Transformer(nn.Module):
    """Full encoder-decoder model. Defaults match the paper's "base" configuration
    (65M parameters at the WMT vocab size)."""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        pad_idx=0,
        tie_weights=False,
        pre_norm=False,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, pre_norm)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, pre_norm)
        # Projects decoder states to vocabulary logits.
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Xavier init (as in the reference implementation) keeps the variance of
        # activations roughly constant through the deep stack.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Embeddings need a different rule. Xavier on a (vocab x d_model) matrix gives
        # std ~ sqrt(2 / (vocab + d_model)) ~ 0.017 for an 8k vocab; even after the
        # sqrt(d_model) scaling in the encoder/decoder that is ~0.27, so the positional
        # encoding (amplitude 1) drowns out the token identity and training stalls.
        # N(0, d_model^-0.5) makes the scaled embedding unit-variance, as in
        # fairseq and Tensor2Tensor. Measured on Multi30k: 3x lower loss at equal steps.
        for embed in (self.encoder.embed, self.decoder.embed):
            nn.init.normal_(embed.weight, mean=0.0, std=d_model ** -0.5)
            with torch.no_grad():
                embed.weight[pad_idx].zero_()

        # Section 3.4: share the target embedding matrix with the output projection.
        # Both map between token identity and d_model space, so sharing them cuts
        # parameters and regularises the model on small datasets.
        if tie_weights:
            self.generator.weight = self.decoder.embed.weight

    def encode(self, src):
        src_mask = make_pad_mask(src, self.pad_idx)
        return self.encoder(src, src_mask), src_mask

    def decode(self, tgt, memory, src_mask):
        # Combine padding mask and causal mask: (B,1,1,T) & (1,1,T,T) -> (B,1,T,T)
        tgt_mask = make_pad_mask(tgt, self.pad_idx) & make_causal_mask(tgt.size(1), tgt.device)
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def forward(self, src, tgt):
        """src: (B, S), tgt: (B, T) -> logits (B, T, tgt_vocab_size)

        `tgt` is the decoder *input* (shifted right, starts with BOS). The loss is
        computed against the same sequence shifted left; see train.py.
        """
        memory, src_mask = self.encode(src)
        out = self.decode(tgt, memory, src_mask)
        return self.generator(out)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, eos_idx, max_len=64, use_cache=True):
        """Autoregressive inference: feed the argmax token back in until EOS.
        src: (B, S) -> (B, <=max_len) generated ids (including BOS).

        use_cache=True runs the decoder on only the newest token each step, reusing
        cached K/V for the prefix (see MultiHeadAttention). Output is identical to the
        uncached path, which re-runs the full prefix and is kept for testing.
        """
        self.eval()
        memory, src_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        caches = self.decoder.new_caches() if use_cache else None
        for step in range(max_len - 1):
            if use_cache:
                # Only the last token goes in; the causal mask is implicit because the
                # cache holds exactly the positions before it.
                out = self.decoder(ys[:, -1:], memory, None, src_mask, caches=caches, offset=step)
            else:
                out = self.decode(ys, memory, src_mask)
            next_tok = self.generator(out[:, -1]).argmax(-1)  # (B,)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok == eos_idx
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def beam_search(self, src, bos_idx, eos_idx, beam_size=4, max_len=64,
                    length_penalty=0.6, use_cache=True):
        """Batched beam search (Section 6.1 of the paper uses beam 4, alpha = 0.6).

        Keeps the `beam_size` most probable partial sequences per sentence instead of
        only the single best (greedy). Finished hypotheses are scored by
        log_prob / ((5 + len) / 6) ** alpha, which stops the search from favouring
        short outputs.

        All B sentences and their K beams are flattened into one batch of B*K rows so
        each step is a single decoder call; the KV cache rows are re-gathered after
        every step to follow the surviving beams. Padding in `src` is handled by the
        source mask, so results do not depend on what a sentence is batched with.

        src: (B, S) -> list of B token-id lists (each starts with BOS and ends with
        EOS unless max_len was hit)
        """
        self.eval()
        B, K, dev = src.size(0), beam_size, src.device
        neg_inf = float("-inf")

        memory, src_mask = self.encode(src)
        memory = memory.repeat_interleave(K, dim=0)      # (B*K, S, D)
        src_mask = src_mask.repeat_interleave(K, dim=0)  # (B*K, 1, 1, S)

        seqs = torch.full((B * K, 1), bos_idx, dtype=torch.long, device=dev)
        # Only beam 0 is live at the start, otherwise K identical copies of BOS would
        # fill the whole beam with duplicates on the first step.
        scores = torch.full((B, K), neg_inf, device=dev)
        scores[:, 0] = 0.0
        finished = [[] for _ in range(B)]
        done = [False] * B
        caches = self.decoder.new_caches() if use_cache else None

        for step in range(max_len - 1):
            if use_cache:
                out = self.decoder(seqs[:, -1:], memory, None, src_mask, caches=caches, offset=step)
            else:
                out = self.decode(seqs, memory, src_mask)
            log_probs = F.log_softmax(self.generator(out[:, -1]), dim=-1)  # (B*K, V)
            V = log_probs.size(-1)

            # Every (beam, next-token) pair is a candidate. Take the top 2K per sentence
            # so that candidates ending in EOS can be retired without starving the beam.
            cand = (scores.view(B * K, 1) + log_probs).view(B, K * V)
            top_scores, top_idx = cand.topk(2 * K, dim=1)
            beam_idx, tok_idx = top_idx // V, top_idx % V
            lp = ((5 + step + 2) / 6) ** length_penalty  # length incl. BOS after this step

            new_scores = torch.full((B, K), neg_inf, device=dev)
            new_parent = torch.arange(B, device=dev).repeat_interleave(K).view(B, K)
            new_tok = torch.full((B, K), eos_idx, dtype=torch.long, device=dev)
            ts, bi, ti = top_scores.tolist(), beam_idx.tolist(), tok_idx.tolist()
            for b in range(B):
                if done[b]:
                    continue
                j = 0
                for c in range(2 * K):
                    if ts[b][c] == neg_inf:
                        break
                    if ti[b][c] == eos_idx:
                        # Only an EOS ranked inside the top K counts as a finished
                        # hypothesis (as in fairseq). Lower-ranked EOS candidates would
                        # fill the finished pool with weak outputs and stop the search
                        # before better, longer hypotheses complete.
                        if c < K and len(finished[b]) < K:
                            finished[b].append((ts[b][c] / lp, seqs[b * K + bi[b][c]].tolist() + [eos_idx]))
                    elif j < K:
                        new_scores[b, j] = ts[b][c]
                        new_parent[b, j] = b * K + bi[b][c]
                        new_tok[b, j] = ti[b][c]
                        j += 1
                    if j == K:
                        break
                if len(finished[b]) >= K:
                    done[b] = True

            flat = new_parent.view(-1)
            seqs = torch.cat([seqs[flat], new_tok.view(-1, 1)], dim=1)
            scores = new_scores
            if use_cache:
                # Self-attention caches must follow their beams; cross-attention K/V
                # are identical across the K beams of a sentence and need no reorder.
                for c in caches:
                    c["self"]["k"] = c["self"]["k"][flat]
                    c["self"]["v"] = c["self"]["v"][flat]
            if all(done):
                break

        results = []
        for b in range(B):
            if not finished[b]:  # hit max_len: fall back to the best live beam
                lp = ((5 + seqs.size(1)) / 6) ** length_penalty
                best = int(scores[b].argmax())
                finished[b].append((scores[b, best].item() / lp, seqs[b * K + best].tolist()))
            results.append(max(finished[b], key=lambda x: x[0])[1])
        return results


if __name__ == "__main__":
    model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
