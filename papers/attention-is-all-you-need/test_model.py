"""
Shape and sanity tests for each Transformer component.
Run with:  python -m pytest test_model.py -v     (or  python test_model.py)
Implement model.py top to bottom; tests are ordered to match.
"""

import torch
import pytest

import model as m

B, S, T, D, H, FF = 2, 7, 5, 64, 4, 128


def test_1_scaled_dot_product_attention():
    q = torch.randn(B, H, T, D // H)
    k = torch.randn(B, H, S, D // H)
    v = torch.randn(B, H, S, D // H)
    out, attn = m.scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, T, D // H)
    assert attn.shape == (B, H, T, S)
    assert torch.allclose(attn.sum(-1), torch.ones(B, H, T), atol=1e-5)

    # masked positions must receive zero weight
    mask = torch.ones(1, 1, T, S, dtype=torch.bool)
    mask[..., -1] = False
    _, attn = m.scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.all(attn[..., -1] == 0)


def test_2_multi_head_attention():
    mha = m.MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    mem = torch.randn(B, S, D)
    assert mha(x, x, x).shape == (B, T, D)          # self-attention
    assert mha(x, mem, mem).shape == (B, T, D)      # cross-attention


def test_3_feed_forward():
    ff = m.PositionwiseFeedForward(D, FF)
    assert ff(torch.randn(B, T, D)).shape == (B, T, D)


def test_4_positional_encoding():
    pe = m.PositionalEncoding(D, dropout=0.0)
    x = torch.zeros(B, T, D)
    out = pe(x)
    assert out.shape == (B, T, D)
    assert torch.allclose(out[0], out[1])                 # same for every batch element
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(D // 2))  # sin(0) = 0
    assert torch.allclose(out[0, 0, 1::2], torch.ones(D // 2))   # cos(0) = 1


def test_5_layers():
    enc = m.EncoderLayer(D, H, FF)
    dec = m.DecoderLayer(D, H, FF)
    src = torch.randn(B, S, D)
    tgt = torch.randn(B, T, D)
    memory = enc(src)
    assert memory.shape == (B, S, D)
    assert dec(tgt, memory).shape == (B, T, D)


def test_6_masks():
    tokens = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
    pad = m.make_pad_mask(tokens, pad_idx=0)
    assert pad.shape == (2, 1, 1, 4)
    assert pad[0, 0, 0].tolist() == [True, True, False, False]

    causal = m.make_causal_mask(3)
    assert causal.shape == (1, 1, 3, 3)
    assert causal[0, 0].tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_6_transformer_forward_and_causality():
    torch.manual_seed(0)
    model = m.Transformer(50, 60, d_model=D, n_layers=2, n_heads=H, d_ff=FF, dropout=0.0)
    model.eval()
    src = torch.randint(1, 50, (B, S))
    tgt = torch.randint(1, 60, (B, T))
    logits = model(src, tgt)
    assert logits.shape == (B, T, 60)

    # changing a future target token must not change earlier logits
    tgt2 = tgt.clone()
    tgt2[:, -1] = (tgt2[:, -1] + 1) % 60
    logits2 = model(src, tgt2)
    assert torch.allclose(logits[:, :-1], logits2[:, :-1], atol=1e-5)


def test_7_pre_norm_variant():
    """pre_norm=True must keep shapes and causality, and add exactly one final
    LayerNorm per stack; pre_norm=False must not."""
    torch.manual_seed(0)
    pre = m.Transformer(50, 60, d_model=D, n_layers=2, n_heads=H, d_ff=FF, dropout=0.0, pre_norm=True)
    post = m.Transformer(50, 60, d_model=D, n_layers=2, n_heads=H, d_ff=FF, dropout=0.0)
    assert isinstance(pre.encoder.final_norm, torch.nn.LayerNorm)
    assert isinstance(post.encoder.final_norm, torch.nn.Identity)
    n_norms = lambda model: sum(isinstance(x, torch.nn.LayerNorm) for x in model.modules())
    assert n_norms(pre) == n_norms(post) + 2

    pre.eval()
    src = torch.randint(1, 50, (B, S))
    tgt = torch.randint(1, 60, (B, T))
    logits = pre(src, tgt)
    assert logits.shape == (B, T, 60)
    tgt2 = tgt.clone()
    tgt2[:, -1] = (tgt2[:, -1] + 1) % 60
    assert torch.allclose(logits[:, :-1], pre(src, tgt2)[:, :-1], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
