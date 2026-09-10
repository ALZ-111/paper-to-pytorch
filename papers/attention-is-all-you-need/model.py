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

    def forward(self, query, key, value, mask=None):
        """
        query: (B, T, D)   key/value: (B, S, D)   mask: broadcastable to (B, H, T, S)
        Returns (B, T, D).

        Self-attention:  query = key = value = x
        Cross-attention: query = decoder state, key = value = encoder output
        """
        q = self._split_heads(self.w_q(query))  # (B, H, T, Dk)
        k = self._split_heads(self.w_k(key))    # (B, H, S, Dk)
        v = self._split_heads(self.w_v(value))  # (B, H, S, Dk)

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

    def forward(self, x):
        """x: (B, T, D) token embeddings -> (B, T, D) with positions added"""
        x = x + self.pe[:, : x.size(1)]
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

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        """x: (B, T, D)  memory: (B, S, D) encoder output -> (B, T, D)"""
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sub2(x, lambda x: self.cross_attn(x, memory, memory, src_mask))
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

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        """tgt: (B, T) token ids, memory: (B, S, D) -> (B, T, D)"""
        x = self.pos(self.embed(tgt) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.final_norm(x)


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
    def greedy_decode(self, src, bos_idx, eos_idx, max_len=64):
        """Autoregressive inference: feed the argmax token back in until EOS.
        src: (B, S) -> (B, <=max_len) generated ids (including BOS)."""
        self.eval()
        memory, src_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            out = self.decode(ys, memory, src_mask)
            next_tok = self.generator(out[:, -1]).argmax(-1)  # (B,)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok == eos_idx
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def beam_search(self, src, bos_idx, eos_idx, beam_size=4, max_len=64, length_penalty=0.6):
        """Beam search decoding for a single source sentence (Section 6.1 of the paper
        uses beam 4 and length penalty alpha = 0.6).

        Keeps the `beam_size` most probable partial sequences at each step instead of
        only the single best (greedy). Finished hypotheses are scored by
        log_prob / ((5 + len) / 6) ** alpha, which stops the search from favouring
        short outputs.

        src: (1, S) -> list of token ids (including BOS, ending with EOS if produced)
        """
        assert src.size(0) == 1, "beam_search decodes one sentence at a time"
        self.eval()
        memory, src_mask = self.encode(src)
        memory = memory.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)

        beams = torch.full((1, 1), bos_idx, dtype=torch.long, device=src.device)
        scores = torch.zeros(1, device=src.device)
        finished = []

        for step in range(max_len - 1):
            k = beams.size(0)
            out = self.decode(beams, memory[:k], src_mask[:k])
            log_probs = F.log_softmax(self.generator(out[:, -1]), dim=-1)  # (k, V)
            vocab = log_probs.size(-1)

            # Every (beam, next-token) pair is a candidate; pick the top beam_size overall.
            cand = (scores.unsqueeze(1) + log_probs).view(-1)
            top_scores, top_idx = cand.topk(min(beam_size, cand.numel()))
            beam_idx, tok_idx = top_idx // vocab, top_idx % vocab
            beams = torch.cat([beams[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            scores = top_scores

            # Move finished hypotheses out of the active set.
            is_eos = tok_idx == eos_idx
            for i in torch.nonzero(is_eos).flatten().tolist():
                lp = ((5 + beams.size(1)) / 6) ** length_penalty
                finished.append((scores[i].item() / lp, beams[i].tolist()))
            keep = ~is_eos
            beams, scores = beams[keep], scores[keep]
            if beams.size(0) == 0 or len(finished) >= beam_size:
                break

        # If nothing finished, fall back to the best unfinished beam.
        if not finished:
            lp = ((5 + beams.size(1)) / 6) ** length_penalty
            finished = [(scores[i].item() / lp, beams[i].tolist()) for i in range(beams.size(0))]
        return max(finished, key=lambda x: x[0])[1]


if __name__ == "__main__":
    model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
