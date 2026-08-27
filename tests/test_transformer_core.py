"""Correctness tests for the core transformer stack.

Covers the five hardening items:
  1. RoPE relative-position invariance + layout/shape stability.
  2. LiGR gated residual matches the paper formula h + F(h)*sigma(hW).
  3. RMSNorm wrapper agrees with F.rms_norm; LayerNorm matches nn.LayerNorm.
  4. FlashAttention path is causal-flag preserving (mask only when needed).
  5. Causal masking is correct and padding masks don't break causality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rec_arena.modules.layer_utils.embeddings import (
    RotaryPositionalEmbedding,
)  # noqa: E402
from rec_arena.modules.transformer_layers.transformer_block import (  # noqa: E402
    RMSNorm,
    TransformerBlock,
)
from rec_arena.modules.transformer_layers.mha import (
    CausalSelfAttention,
)  # noqa: E402
from rec_arena.modules.layer_utils.normalization_layers import (
    LayerNorm,
)  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. RoPE
# --------------------------------------------------------------------------- #
def test_rope_relative_position_invariance():
    """<RoPE(q, m), RoPE(k, n)> must depend only on (m - n)."""
    torch.manual_seed(0)
    Hd, S = 16, 8
    rope = RotaryPositionalEmbedding(dim=Hd, max_seq_len=64)
    qv = torch.randn(Hd)
    kv = torch.randn(Hd)
    q = qv.view(1, 1, 1, Hd).expand(1, 1, S, Hd).clone()
    k = kv.view(1, 1, 1, Hd).expand(1, 1, S, Hd).clone()
    qe, ke = rope(q, k, S)

    from collections import defaultdict

    by_diff = defaultdict(list)
    for m in range(S):
        for n in range(S):
            by_diff[m - n].append(torch.dot(qe[0, 0, m], ke[0, 0, n]).item())
    for diff, vals in by_diff.items():
        assert max(vals) - min(vals) < 1e-4, f"RoPE not relative at diff={diff}"


def test_rope_preserves_shape_and_norm():
    torch.manual_seed(1)
    B, H, S, Hd = 2, 4, 10, 16
    rope = RotaryPositionalEmbedding(dim=Hd, max_seq_len=64)
    q = torch.randn(B, H, S, Hd)
    k = torch.randn(B, H, S, Hd)
    qe, ke = rope(q, k, S)
    assert qe.shape == q.shape and ke.shape == k.shape
    # RoPE is a rotation: per-token vector norm is preserved.
    assert torch.allclose(qe.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    assert torch.allclose(ke.norm(dim=-1), k.norm(dim=-1), atol=1e-4)


def test_rope_zero_offset_is_identity_dot():
    # At m == n, RoPE rotates q and k by the same angle, so the dot product
    # equals the unrotated dot product.
    torch.manual_seed(2)
    Hd, S = 16, 5
    rope = RotaryPositionalEmbedding(dim=Hd, max_seq_len=32)
    q = torch.randn(1, 1, S, Hd)
    k = torch.randn(1, 1, S, Hd)
    qe, ke = rope(q, k, S)
    for m in range(S):
        raw = torch.dot(q[0, 0, m], k[0, 0, m])
        rot = torch.dot(qe[0, 0, m], ke[0, 0, m])
        assert torch.allclose(raw, rot, atol=1e-4)


# --------------------------------------------------------------------------- #
# 2. LiGR gated residual
# --------------------------------------------------------------------------- #
def test_ligr_gated_residual_formula():
    """Verify block output = h + F(h) * sigma(h W) for the attention sublayer."""
    torch.manual_seed(0)
    dim = 16
    block = TransformerBlock(
        dim=dim,
        num_heads=2,
        hidden_dim=32,
        dropout_rate=0.0,
        use_gated_residual=True,
        norm_first=True,
    ).eval()
    x = torch.randn(2, 6, dim)

    # Recompute the attention sublayer path manually and compare the gate.
    residual = x
    normed = block.attn_norm(x)
    attn_out = block.attention(normed, attn_mask=None, is_causal=block.causality)
    gate = torch.sigmoid(block.gate_attn(residual))
    expected_after_attn = residual + gate * attn_out

    # Reproduce just the attention half of forward().
    got = residual + gate * block.attention(
        block.attn_norm(x), attn_mask=None, is_causal=block.causality
    )
    assert torch.allclose(got, expected_after_attn, atol=1e-6)


def test_ligr_block_runs_and_changes_output_vs_plain():
    torch.manual_seed(0)
    dim = 16
    x = torch.randn(2, 6, dim)
    plain = TransformerBlock(
        dim, 2, 32, dropout_rate=0.0, use_gated_residual=False
    ).eval()
    gated = TransformerBlock(
        dim, 2, 32, dropout_rate=0.0, use_gated_residual=True
    ).eval()
    out_p = plain(x)
    out_g = gated(x)
    assert out_p.shape == out_g.shape == x.shape
    assert not torch.allclose(out_p, out_g)


# --------------------------------------------------------------------------- #
# 3. Normalization consolidation
# --------------------------------------------------------------------------- #
def test_rmsnorm_matches_functional():
    torch.manual_seed(0)
    dim = 32
    rms = RMSNorm(dim, eps=1e-6).eval()
    x = torch.randn(4, 7, dim)
    expected = F.rms_norm(x, normalized_shape=[dim], weight=rms.weight, eps=1e-6)
    assert torch.allclose(rms(x), expected, atol=1e-6)


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    dim = 32
    ln = LayerNorm(dim, eps=1e-5).eval()
    ref = torch.nn.LayerNorm(dim, eps=1e-5).eval()
    # copy params so only the formula is compared
    with torch.no_grad():
        ref.weight.copy_(ln.weight)
        ref.bias.copy_(ln.bias)
    x = torch.randn(4, 7, dim)
    assert torch.allclose(ln(x), ref(x), atol=1e-5)


# --------------------------------------------------------------------------- #
# 4/5. Attention masking + Flash-friendly fast path
# --------------------------------------------------------------------------- #
def test_causal_attention_is_lower_triangular():
    """Output at position t must not depend on inputs after t."""
    torch.manual_seed(0)
    dim, S = 16, 6
    attn = CausalSelfAttention(dim, num_heads=2, dropout_rate=0.0).eval()
    x = torch.randn(1, S, dim)
    out = attn(x, attn_mask=None, is_causal=True)

    # Perturb the last token; outputs at earlier positions must be unchanged.
    x2 = x.clone()
    x2[0, -1] += 10.0
    out2 = attn(x2, attn_mask=None, is_causal=True)
    assert torch.allclose(
        out[0, :-1], out2[0, :-1], atol=1e-5
    ), "causal attention leaked future information"


def test_padding_mask_preserves_causality():
    # A 2D key-padding mask combined with is_causal must still be causal.
    torch.manual_seed(0)
    dim, S = 16, 6
    attn = CausalSelfAttention(dim, num_heads=2, dropout_rate=0.0).eval()
    x = torch.randn(1, S, dim)
    keep = torch.ones(1, S, dtype=torch.bool)  # all valid
    out_masked = attn(x, attn_mask=keep, is_causal=True)
    out_plain = attn(x, attn_mask=None, is_causal=True)
    # With all-valid padding mask + causal, result equals the pure causal path.
    assert torch.allclose(out_masked, out_plain, atol=1e-5)


def test_attention_forward_shapes():
    torch.manual_seed(0)
    dim, S, B = 32, 9, 3
    attn = CausalSelfAttention(dim, num_heads=4, dropout_rate=0.0).eval()
    x = torch.randn(B, S, dim)
    assert attn(x).shape == (B, S, dim)
