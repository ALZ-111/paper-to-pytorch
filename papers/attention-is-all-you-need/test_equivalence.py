"""
Numerical equivalence against torch.nn.Transformer.

The strongest correctness check available: copy this implementation's weights into
PyTorch's reference nn.Transformer (same post-norm layout, dropout off) and require
the outputs to match to float32 precision. If any mask, projection, head split, or
residual were wired differently, the outputs would diverge.

Run with:  python -m pytest test_equivalence.py -v
"""

import torch
import torch.nn as nn

import model as m

torch.manual_seed(0)
B, S, T, D, H, FF, L = 3, 9, 6, 32, 4, 64, 2
PAD = 0


def _copy_mha(mine: m.MultiHeadAttention, theirs: nn.MultiheadAttention):
    """Ours keeps W_Q, W_K, W_V as separate Linear layers; torch packs them into one
    (3D x D) in_proj_weight stacked in q, k, v order."""
    with torch.no_grad():
        theirs.in_proj_weight.copy_(torch.cat([mine.w_q.weight, mine.w_k.weight, mine.w_v.weight]))
        theirs.in_proj_bias.copy_(torch.cat([mine.w_q.bias, mine.w_k.bias, mine.w_v.bias]))
        theirs.out_proj.weight.copy_(mine.w_o.weight)
        theirs.out_proj.bias.copy_(mine.w_o.bias)


def _copy_ff_and_norms(mine, theirs, norms):
    with torch.no_grad():
        theirs.linear1.weight.copy_(mine.ff.w_1.weight)
        theirs.linear1.bias.copy_(mine.ff.w_1.bias)
        theirs.linear2.weight.copy_(mine.ff.w_2.weight)
        theirs.linear2.bias.copy_(mine.ff.w_2.bias)
        for mine_sub, their_norm in norms:
            their_norm.weight.copy_(mine_sub.norm.weight)
            their_norm.bias.copy_(mine_sub.norm.bias)


def _randomise_norms(module):
    """Default LayerNorm weights are all-ones/zeros, which would hide a mis-mapped norm.
    Give them random affine parameters so the test is sensitive to the mapping."""
    for sub in module.modules():
        if isinstance(sub, nn.LayerNorm):
            with torch.no_grad():
                sub.weight.uniform_(0.5, 1.5)
                sub.bias.uniform_(-0.5, 0.5)


def build_pair():
    torch_model = nn.Transformer(
        d_model=D, nhead=H, num_encoder_layers=L, num_decoder_layers=L,
        dim_feedforward=FF, dropout=0.0, batch_first=True, norm_first=False,
    )
    # nn.Transformer appends an extra LayerNorm after each stack; the paper (and this
    # implementation) do not, so remove them for a like-for-like comparison.
    torch_model.encoder.norm = nn.Identity()
    torch_model.decoder.norm = nn.Identity()

    enc_layers = nn.ModuleList([m.EncoderLayer(D, H, FF, dropout=0.0) for _ in range(L)])
    dec_layers = nn.ModuleList([m.DecoderLayer(D, H, FF, dropout=0.0) for _ in range(L)])
    _randomise_norms(enc_layers)
    _randomise_norms(dec_layers)

    for mine, theirs in zip(enc_layers, torch_model.encoder.layers):
        _copy_mha(mine.self_attn, theirs.self_attn)
        _copy_ff_and_norms(mine, theirs, [(mine.sub1, theirs.norm1), (mine.sub2, theirs.norm2)])
    for mine, theirs in zip(dec_layers, torch_model.decoder.layers):
        _copy_mha(mine.self_attn, theirs.self_attn)
        _copy_mha(mine.cross_attn, theirs.multihead_attn)
        _copy_ff_and_norms(mine, theirs, [(mine.sub1, theirs.norm1), (mine.sub2, theirs.norm2),
                                          (mine.sub3, theirs.norm3)])
    torch_model.eval()
    enc_layers.eval()
    dec_layers.eval()
    return enc_layers, dec_layers, torch_model


def make_inputs():
    src = torch.randn(B, S, D)
    tgt = torch.randn(B, T, D)
    # Padding at the end of some sequences, as in real batches.
    src_tokens = torch.ones(B, S, dtype=torch.long)
    src_tokens[0, -2:] = PAD
    src_tokens[1, -1:] = PAD
    tgt_tokens = torch.ones(B, T, dtype=torch.long)
    tgt_tokens[2, -1:] = PAD
    return src, tgt, src_tokens, tgt_tokens


def run_mine(enc_layers, dec_layers, src, tgt, src_tokens, tgt_tokens):
    src_mask = m.make_pad_mask(src_tokens, PAD)
    tgt_mask = m.make_pad_mask(tgt_tokens, PAD) & m.make_causal_mask(T)
    x = src
    for layer in enc_layers:
        x = layer(x, src_mask)
    memory = x
    y = tgt
    for layer in dec_layers:
        y = layer(y, memory, tgt_mask, src_mask)
    return memory, y


def run_theirs(torch_model, src, tgt, src_tokens, tgt_tokens):
    # torch convention is the inverse of ours: True = *ignore* this key.
    src_kpm = src_tokens == PAD
    tgt_kpm = tgt_tokens == PAD
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # True = blocked
    memory = torch_model.encoder(src, src_key_padding_mask=src_kpm)
    out = torch_model.decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_kpm,
                              memory_key_padding_mask=src_kpm)
    return memory, out


def test_encoder_matches_torch():
    enc, dec, ref = build_pair()
    src, tgt, st, tt = make_inputs()
    mine, _ = run_mine(enc, dec, src, tgt, st, tt)
    theirs, _ = run_theirs(ref, src, tgt, st, tt)
    # Compare non-padding positions only: torch's fast path may leave padded rows undefined.
    keep = st != PAD
    assert torch.allclose(mine[keep], theirs[keep], atol=1e-5), (mine - theirs).abs().max()


def test_decoder_matches_torch():
    enc, dec, ref = build_pair()
    src, tgt, st, tt = make_inputs()
    _, mine = run_mine(enc, dec, src, tgt, st, tt)
    _, theirs = run_theirs(ref, src, tgt, st, tt)
    keep = tt != PAD
    assert torch.allclose(mine[keep], theirs[keep], atol=1e-5), (mine - theirs).abs().max()


def test_gradients_match_torch():
    """Same forward implies same backward only if autograd sees the same graph;
    check the gradient w.r.t. the decoder input agrees too."""
    enc, dec, ref = build_pair()
    src, tgt, st, tt = make_inputs()
    tgt_a = tgt.clone().requires_grad_(True)
    tgt_b = tgt.clone().requires_grad_(True)
    keep = tt != PAD
    _, out_a = run_mine(enc, dec, src, tgt_a, st, tt)
    _, out_b = run_theirs(ref, src, tgt_b, st, tt)
    out_a[keep].sum().backward()
    out_b[keep].sum().backward()
    assert torch.allclose(tgt_a.grad, tgt_b.grad, atol=1e-5), (tgt_a.grad - tgt_b.grad).abs().max()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
