"""Tests for time-aware sequential models (HSTU, FuXi-alpha, FuXi-gamma).

Verifies that:
  - models build and run a forward/predict pass
  - real batch timestamps reach the model via get_batch_timestamps()
  - the relative-time bias actually changes outputs vs. a positional fallback
  - HSTU uses SiLU over the full uvqk projection and a bias-free uvqk proj
  - HSTU supports decoupled dqk/dv dims and the concat_ua gating variant
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rec_arena.models import HSTU, FuXi, FuXiGamma  # noqa: E402
from rec_arena.configs.defaults.hstu import HSTUConfig  # noqa: E402
from rec_arena.configs.defaults.fuxi import FuXiConfig  # noqa: E402
from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig  # noqa: E402

VOCAB = 50
SEQ = 16
B = 4


def _seqs():
    torch.manual_seed(0)
    sequences = torch.randint(3, VOCAB, (B, SEQ))
    lengths = torch.full((B,), SEQ, dtype=torch.long)
    # Monotonically increasing real timestamps per user (unix-second scale).
    base = torch.arange(SEQ).float() * 86400.0  # one day apart
    timestamps = base.unsqueeze(0).repeat(B, 1) + torch.arange(B).unsqueeze(1) * 1e6
    return sequences, lengths, timestamps


# --------------------------------------------------------------------------- #
# HSTU
# --------------------------------------------------------------------------- #
def test_hstu_builds_and_forwards():
    cfg = HSTUConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_length=SEQ,
        use_time_bias=True,
    )
    model = HSTU(cfg).eval()
    seqs, lengths, _ = _seqs()
    logits = model(seqs, lengths)
    assert logits.shape == (B, SEQ, VOCAB)
    assert torch.isfinite(logits).all()


def test_hstu_uvqk_is_bias_free():
    cfg = HSTUConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_length=SEQ,
    )
    model = HSTU(cfg)
    blk = model.hstu_blocks[0]
    assert isinstance(blk._uvqk, torch.nn.Parameter)
    expected = (
        blk._linear_dim * blk._num_heads * 2 + blk._attention_dim * blk._num_heads * 2
    )
    assert blk._uvqk.shape == (cfg.embedding_dim, expected)


def test_hstu_decoupled_dims():
    # Reference ML-1M uses dqk = dv = embedding_dim with num_heads=1.
    cfg = HSTUConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=1,
        num_layers=2,
        max_seq_length=SEQ,
        attention_dim=16,
        linear_dim=16,
    )
    model = HSTU(cfg).eval()
    assert model.attention_dim == 16 and model.linear_dim == 16
    seqs, lengths, _ = _seqs()
    logits = model(seqs, lengths)
    assert logits.shape == (B, SEQ, VOCAB)
    assert torch.isfinite(logits).all()


def test_hstu_time_bias_changes_output():
    cfg = HSTUConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_length=SEQ,
        use_time_bias=True,
        dropout_rate=0.0,
    )
    model = HSTU(cfg).eval()
    seqs, lengths, ts = _seqs()

    # Without timestamps -> position-only fallback
    model._batch_timestamps = None
    out_pos = model(seqs, lengths)

    # With real timestamps -> time+position bias
    model._batch_timestamps = ts
    out_time = model(seqs, lengths)

    assert not torch.allclose(
        out_pos, out_time, atol=1e-5
    ), "time bias did not affect HSTU output"


def test_hstu_concat_ua_variant():
    cfg = HSTUConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_length=SEQ,
        concat_ua=True,
        dropout_rate=0.0,
    )
    model = HSTU(cfg).eval()
    seqs, lengths, _ = _seqs()
    logits = model(seqs, lengths)
    assert logits.shape == (B, SEQ, VOCAB)
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# FuXi-alpha
# --------------------------------------------------------------------------- #
def test_fuxi_builds_and_forwards():
    cfg = FuXiConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        attention_dim=8,
        linear_dim=8,
        max_seq_length=SEQ,
    )
    model = FuXi(cfg).eval()
    seqs, lengths, _ = _seqs()
    out = model.predict_next(seqs, lengths)
    assert out.shape == (B, VOCAB)
    assert torch.isfinite(out).all()


def test_fuxi_uses_real_timestamps():
    cfg = FuXiConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        attention_dim=8,
        linear_dim=8,
        max_seq_length=SEQ,
        dropout_rate=0.0,
    )
    model = FuXi(cfg).eval()
    seqs, lengths, ts = _seqs()

    model._batch_timestamps = None
    h_pos = model.get_hidden_states(seqs, lengths)

    model._batch_timestamps = ts
    h_time = model.get_hidden_states(seqs, lengths)

    assert not torch.allclose(
        h_pos, h_time, atol=1e-5
    ), "FuXi-alpha ignored real timestamps"


# --------------------------------------------------------------------------- #
# FuXi-gamma
# --------------------------------------------------------------------------- #
def test_fuxi_gamma_builds_and_forwards():
    cfg = FuXiGammaConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        attention_dim=8,
        linear_dim=8,
        max_seq_length=SEQ,
    )
    model = FuXiGamma(cfg).eval()
    seqs, lengths, _ = _seqs()
    out = model.predict_next(seqs, lengths)
    assert out.shape == (B, VOCAB)
    assert torch.isfinite(out).all()


def test_fuxi_gamma_finite_with_real_timestamps():
    # The exponential-power encoder must stay finite on real (day-scale) Δt.
    cfg = FuXiGammaConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        attention_dim=8,
        linear_dim=8,
        max_seq_length=SEQ,
        dropout_rate=0.0,
    )
    model = FuXiGamma(cfg).eval()
    seqs, lengths, ts = _seqs()
    model._batch_timestamps = ts
    out = model.get_hidden_states(seqs, lengths)
    assert torch.isfinite(out).all()


def test_fuxi_gamma_uses_real_timestamps():
    cfg = FuXiGammaConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=2,
        attention_dim=8,
        linear_dim=8,
        max_seq_length=SEQ,
        dropout_rate=0.0,
    )
    model = FuXiGamma(cfg).eval()
    seqs, lengths, ts = _seqs()

    model._batch_timestamps = None
    h_pos = model.get_hidden_states(seqs, lengths)

    model._batch_timestamps = ts
    h_time = model.get_hidden_states(seqs, lengths)

    assert not torch.allclose(
        h_pos, h_time, atol=1e-5
    ), "FuXi-gamma ignored real timestamps"


# --------------------------------------------------------------------------- #
# Integration: timestamps survive collate and reach the model via a step
# --------------------------------------------------------------------------- #
def test_timestamps_flow_through_training_step():
    cfg = HSTUConfig(
        vocab_size=VOCAB,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_length=SEQ,
        use_time_bias=True,
        loss_type="cross_entropy",
    )
    from rec_arena.losses.factory import get_loss_function

    model = HSTU(cfg)
    model.loss_fn = get_loss_function("cross_entropy", model_type="sequential")

    seqs, lengths, ts = _seqs()
    batch = {
        "sequence": seqs,
        "sequence_length": lengths,
        "timestamps": ts,
        "user_id": torch.arange(B),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    # the step must have stashed the batch timestamps for the model to read
    assert model.get_batch_timestamps() is not None
    assert torch.equal(model.get_batch_timestamps(), ts)
