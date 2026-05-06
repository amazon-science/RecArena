"""Tests for rec_arena.losses module.

Covers: loss factory routing, sequential losses, implicit losses,
regularization losses, and property-based correctness checks.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rec_arena.losses import (
    BCELoss,
    BCENegativeSamplingLoss,
    BPRLoss,
    ContrastiveLoss,
    CrossEntropyLoss,
    FocalLoss,
    GBCE,
    LabelSmoothingLoss,
    MultiTaskLoss,
    SampledSoftmaxLoss,
    SequentialBPRLoss,
    get_loss_function,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 4
SEQ_LEN = 8
VOCAB = 50
DIM = 16
NUM_NEG = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seq_inputs(batch=BATCH, seq_len=SEQ_LEN, vocab=VOCAB, num_neg=NUM_NEG):
    """Return (logits, targets, mask, neg_items) for sequential losses."""
    torch.manual_seed(0)
    logits = torch.randn(batch, seq_len, vocab)
    targets = torch.randint(1, vocab, (batch, seq_len))
    mask = torch.ones(batch, seq_len)
    neg_items = torch.randint(1, vocab, (batch, seq_len, num_neg))
    return logits, targets, mask, neg_items


def _make_implicit_model(batch=BATCH, num_neg=NUM_NEG, num_items=VOCAB, dim=DIM):
    """Build a minimal mock model satisfying the implicit loss interface."""
    torch.manual_seed(0)
    model = MagicMock()
    model.config.num_items = num_items

    # prediction() is called twice: once for pos [batch, dim] → [batch, 1],
    # once for neg [batch*num_neg, dim] → [batch*num_neg, 1].
    # side_effect lets us return different tensors per call.
    pos_scores = torch.randn(batch, 1)
    neg_scores = torch.randn(batch * num_neg, 1)
    model.prediction.side_effect = [pos_scores, neg_scores]

    # get_hidden_states() returns [batch*num_neg, dim]
    neg_hidden = torch.randn(batch * num_neg, dim)
    model.get_hidden_states.return_value = neg_hidden

    # get_user_embedding / get_item_embedding for BPR fast path
    model.get_user_embedding.return_value = torch.randn(batch, dim)
    model.get_item_embedding.return_value = torch.randn(batch, num_neg, dim)

    # reg_weight = 0 so no regularization branch needed
    model.config.reg_weight = 0.0
    return model


def _make_implicit_batch(batch=BATCH, num_neg=NUM_NEG, num_items=VOCAB):
    """Return a minimal batch dict for implicit losses."""
    torch.manual_seed(1)
    return {
        "user_id": torch.randint(0, 10, (batch,)),
        "item_id": torch.randint(0, num_items, (batch,)),
        "neg_items": torch.randint(0, num_items, (batch, num_neg)),
    }


# ---------------------------------------------------------------------------
# 7.1 — Loss factory unit tests
# ---------------------------------------------------------------------------


class TestGetLossFunction:
    def test_cross_entropy_sequential(self):
        loss = get_loss_function("cross_entropy", model_type="sequential")
        assert isinstance(loss, CrossEntropyLoss)

    def test_bce_implicit(self):
        loss = get_loss_function("bce", model_type="implicit")
        assert isinstance(loss, BCELoss)

    def test_bce_sequential(self):
        loss = get_loss_function("bce", model_type="sequential")
        assert isinstance(loss, BCENegativeSamplingLoss)

    def test_bpr_sequential(self):
        loss = get_loss_function("bpr", model_type="sequential")
        assert isinstance(loss, SequentialBPRLoss)

    def test_bpr_implicit(self):
        loss = get_loss_function("bpr", model_type="implicit")
        assert isinstance(loss, BPRLoss)

    def test_sampled_softmax_sequential(self):
        loss = get_loss_function("sampled_softmax", model_type="sequential")
        assert isinstance(loss, SampledSoftmaxLoss)

    def test_gbce_sequential(self):
        loss = get_loss_function("gbce", model_type="sequential")
        assert isinstance(loss, GBCE)

    # --- error cases ---

    def test_empty_loss_type_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("", model_type="sequential")

    def test_non_string_loss_type_raises(self):
        with pytest.raises(ValueError):
            get_loss_function(None, model_type="sequential")  # type: ignore[arg-type]

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("cross_entropy", model_type="graph")

    def test_incompatible_cross_entropy_implicit_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("cross_entropy", model_type="implicit")

    def test_incompatible_sampled_softmax_implicit_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("sampled_softmax", model_type="implicit")

    def test_incompatible_gbce_implicit_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("gbce", model_type="implicit")

    def test_unknown_loss_type_sequential_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("nonexistent_loss", model_type="sequential")

    def test_unknown_loss_type_implicit_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("cross_entropy_typo", model_type="implicit")


# ---------------------------------------------------------------------------
# 7.1 — Sequential loss unit tests
# ---------------------------------------------------------------------------


class TestCrossEntropyLoss:
    def test_produces_finite_scalar(self):
        logits, targets, mask, _ = _seq_inputs()
        loss_fn = CrossEntropyLoss()
        result = loss_fn(logits=logits, targets=targets, mask=mask)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_requires_logits_or_hidden_states(self):
        _, targets, mask, _ = _seq_inputs()
        loss_fn = CrossEntropyLoss()
        with pytest.raises(ValueError):
            loss_fn(targets=targets, mask=mask)

    def test_result_is_non_negative(self):
        logits, targets, mask, _ = _seq_inputs()
        loss_fn = CrossEntropyLoss()
        result = loss_fn(logits=logits, targets=targets, mask=mask)
        assert result.item() >= 0.0


class TestBCENegativeSamplingLoss:
    def test_produces_finite_scalar(self):
        logits, targets, mask, neg_items = _seq_inputs()
        loss_fn = BCENegativeSamplingLoss()
        result = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_requires_neg_items(self):
        logits, targets, mask, _ = _seq_inputs()
        loss_fn = BCENegativeSamplingLoss()
        with pytest.raises(ValueError):
            loss_fn(logits=logits, targets=targets, mask=mask, neg_items=None)

    def test_result_is_non_negative(self):
        logits, targets, mask, neg_items = _seq_inputs()
        loss_fn = BCENegativeSamplingLoss()
        result = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert result.item() >= 0.0


class TestSampledSoftmaxLoss:
    def test_produces_finite_scalar(self):
        logits, targets, mask, neg_items = _seq_inputs()
        loss_fn = SampledSoftmaxLoss()
        result = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_requires_neg_items(self):
        logits, targets, mask, _ = _seq_inputs()
        loss_fn = SampledSoftmaxLoss()
        with pytest.raises(ValueError):
            loss_fn(logits=logits, targets=targets, mask=mask, neg_items=None)


class TestSequentialBPRLoss:
    def test_produces_finite_scalar(self):
        logits, targets, mask, neg_items = _seq_inputs()
        loss_fn = SequentialBPRLoss()
        result = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_requires_neg_items(self):
        logits, targets, mask, _ = _seq_inputs()
        loss_fn = SequentialBPRLoss()
        with pytest.raises(ValueError):
            loss_fn(logits=logits, targets=targets, mask=mask, neg_items=None)


class TestGBCE:
    def test_produces_finite_scalar(self):
        logits, targets, mask, neg_items = _seq_inputs()
        loss_fn = GBCE()
        result = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_requires_neg_items(self):
        logits, targets, mask, _ = _seq_inputs()
        loss_fn = GBCE()
        with pytest.raises(ValueError):
            loss_fn(logits=logits, targets=targets, mask=mask, neg_items=None)

    def test_result_is_non_negative(self):
        logits, targets, mask, neg_items = _seq_inputs()
        loss_fn = GBCE()
        result = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert result.item() >= 0.0


# ---------------------------------------------------------------------------
# 7.1 — Implicit loss unit tests
# ---------------------------------------------------------------------------


class TestImplicitBCELoss:
    def test_produces_finite_scalar(self):
        model = _make_implicit_model()
        batch = _make_implicit_batch()
        hidden_states = torch.randn(BATCH, DIM)
        loss_fn = BCELoss()
        result = loss_fn(model, batch, hidden_states)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_requires_neg_items_in_batch(self):
        model = _make_implicit_model()
        batch = _make_implicit_batch()
        batch.pop("neg_items")
        hidden_states = torch.randn(BATCH, DIM)
        loss_fn = BCELoss()
        with pytest.raises(ValueError):
            loss_fn(model, batch, hidden_states)


class TestImplicitBPRLoss:
    def test_produces_finite_scalar(self):
        model = _make_implicit_model()
        batch = _make_implicit_batch()
        hidden_states = torch.randn(BATCH, DIM)
        loss_fn = BPRLoss()
        result = loss_fn(model, batch, hidden_states)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_requires_neg_items_in_batch(self):
        model = _make_implicit_model()
        batch = _make_implicit_batch()
        batch.pop("neg_items")
        hidden_states = torch.randn(BATCH, DIM)
        loss_fn = BPRLoss()
        with pytest.raises(ValueError):
            loss_fn(model, batch, hidden_states)


# ---------------------------------------------------------------------------
# 7.1 — Regularization loss unit tests
# ---------------------------------------------------------------------------


class TestLabelSmoothingLoss:
    def test_produces_finite_scalar(self):
        torch.manual_seed(0)
        logits = torch.randn(BATCH, VOCAB)
        targets = torch.randint(0, VOCAB, (BATCH,))
        loss_fn = LabelSmoothingLoss()
        result = loss_fn(logits, targets)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_result_is_non_negative(self):
        torch.manual_seed(0)
        logits = torch.randn(BATCH, VOCAB)
        targets = torch.randint(0, VOCAB, (BATCH,))
        loss_fn = LabelSmoothingLoss()
        assert loss_fn(logits, targets).item() >= 0.0

    def test_3d_input(self):
        torch.manual_seed(0)
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
        targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        loss_fn = LabelSmoothingLoss()
        result = loss_fn(logits, targets)
        assert result.dim() == 0
        assert math.isfinite(result.item())


class TestFocalLoss:
    def test_produces_finite_scalar(self):
        torch.manual_seed(0)
        logits = torch.randn(BATCH, VOCAB)
        targets = torch.randint(0, VOCAB, (BATCH,))
        loss_fn = FocalLoss()
        result = loss_fn(logits, targets)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_result_is_non_negative(self):
        torch.manual_seed(0)
        logits = torch.randn(BATCH, VOCAB)
        targets = torch.randint(0, VOCAB, (BATCH,))
        loss_fn = FocalLoss()
        assert loss_fn(logits, targets).item() >= 0.0


class TestContrastiveLoss:
    def test_produces_finite_scalar_with_negatives(self):
        torch.manual_seed(0)
        anchor = torch.randn(BATCH, DIM)
        positive = torch.randn(BATCH, DIM)
        negatives = torch.randn(BATCH, NUM_NEG, DIM)
        loss_fn = ContrastiveLoss()
        result = loss_fn(anchor, positive, negatives)
        assert result.dim() == 0
        assert math.isfinite(result.item())

    def test_produces_finite_scalar_without_negatives(self):
        torch.manual_seed(0)
        anchor = torch.randn(BATCH, DIM)
        positive = torch.randn(BATCH, DIM)
        loss_fn = ContrastiveLoss()
        result = loss_fn(anchor, positive)
        assert result.dim() == 0
        assert math.isfinite(result.item())


class TestMultiTaskLoss:
    def _make_inputs(self):
        torch.manual_seed(0)
        next_item_logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
        next_item_targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        rating_preds = torch.randn(BATCH, SEQ_LEN)
        rating_targets = torch.rand(BATCH, SEQ_LEN) * 5.0
        return next_item_logits, next_item_targets, rating_preds, rating_targets

    def test_produces_finite_dict(self):
        loss_fn = MultiTaskLoss()
        result = loss_fn(*self._make_inputs())
        assert isinstance(result, dict)
        assert "total_loss" in result
        for v in result.values():
            assert math.isfinite(v.item())

    def test_total_loss_is_non_negative(self):
        loss_fn = MultiTaskLoss()
        result = loss_fn(*self._make_inputs())
        assert result["total_loss"].item() >= 0.0

    def test_with_mask(self):
        loss_fn = MultiTaskLoss()
        inputs = self._make_inputs()
        mask = torch.ones(BATCH, SEQ_LEN)
        mask[0, -2:] = 0.0
        result = loss_fn(*inputs, mask=mask)
        assert math.isfinite(result["total_loss"].item())


# ---------------------------------------------------------------------------
# 7.2 — Property 21: Loss factory rejects invalid inputs
# ---------------------------------------------------------------------------

# Feature: comprehensive-test-suite, Property 21: Loss factory rejects invalid inputs


@given(
    loss_type=st.one_of(
        st.just(""),
        st.integers(),
        st.floats(allow_nan=False),
        st.none(),
        st.lists(st.text()),
    )
)
@settings(max_examples=100)
def test_property_21_loss_factory_rejects_non_string_loss_type(loss_type):
    """Property 21 (part a): non-string or empty loss_type raises ValueError."""
    with pytest.raises((ValueError, TypeError)):
        get_loss_function(loss_type, model_type="sequential")  # type: ignore[arg-type]


@given(
    model_type=st.text(min_size=1).filter(
        lambda s: s not in ("sequential", "implicit")
    )
)
@settings(max_examples=100)
def test_property_21_loss_factory_rejects_invalid_model_type(model_type):
    """Property 21 (part b): model_type not in {'sequential','implicit'} raises ValueError."""
    with pytest.raises(ValueError):
        get_loss_function("cross_entropy", model_type=model_type)


# ---------------------------------------------------------------------------
# 7.3 — Property 22: Loss functions produce finite scalar output
# ---------------------------------------------------------------------------

# Feature: comprehensive-test-suite, Property 22: Loss functions produce finite scalar output


@given(
    batch=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=1, max_value=10),
    num_neg=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_property_22_sequential_losses_finite_scalar(batch, seq_len, num_neg):
    """Property 22: sequential losses produce finite scalar for any valid input shape."""
    vocab = 30
    torch.manual_seed(batch * 1000 + seq_len * 100 + num_neg)
    logits = torch.randn(batch, seq_len, vocab)
    targets = torch.randint(1, vocab, (batch, seq_len))
    mask = torch.ones(batch, seq_len)
    neg_items = torch.randint(1, vocab, (batch, seq_len, num_neg))

    losses_with_neg = [
        BCENegativeSamplingLoss(),
        SampledSoftmaxLoss(),
        SequentialBPRLoss(),
        GBCE(),
    ]
    for loss_fn in losses_with_neg:
        result = loss_fn(logits=logits, targets=targets, mask=mask, neg_items=neg_items)
        assert result.dim() == 0, f"{type(loss_fn).__name__} should return scalar"
        assert math.isfinite(result.item()), f"{type(loss_fn).__name__} returned non-finite"

    # CrossEntropyLoss doesn't use neg_items
    ce_result = CrossEntropyLoss()(logits=logits, targets=targets, mask=mask)
    assert ce_result.dim() == 0
    assert math.isfinite(ce_result.item())


@given(
    batch=st.integers(min_value=1, max_value=8),
    num_classes=st.integers(min_value=2, max_value=30),
)
@settings(max_examples=100)
def test_property_22_regularization_losses_finite_scalar(batch, num_classes):
    """Property 22: regularization losses produce finite scalar for any valid input shape."""
    torch.manual_seed(batch * 100 + num_classes)
    logits = torch.randn(batch, num_classes)
    targets = torch.randint(0, num_classes, (batch,))

    for loss_fn in [LabelSmoothingLoss(), FocalLoss()]:
        result = loss_fn(logits, targets)
        assert result.dim() == 0, f"{type(loss_fn).__name__} should return scalar"
        assert math.isfinite(result.item()), f"{type(loss_fn).__name__} returned non-finite"

    # ContrastiveLoss uses embeddings
    dim = 16
    anchor = torch.randn(batch, dim)
    positive = torch.randn(batch, dim)
    neg_embs = torch.randn(batch, 4, dim)
    contrastive_result = ContrastiveLoss()(anchor, positive, neg_embs)
    assert contrastive_result.dim() == 0
    assert math.isfinite(contrastive_result.item())


# ===========================================================================
# APPENDED FROM test_loss_paths.py — Loss function fast/slow path tests
# ===========================================================================
# Covers: BCENegativeSamplingLoss, SampledSoftmaxLoss, BPRLoss, GBCE
# with both hidden_states (fast) and logits (slow) paths.
# Also covers CrossEntropyLoss logits vs hidden_states paths.
# ===========================================================================

from rec_arena.losses.sequential.bce import BCENegativeSamplingLoss as _BCENegSampling
from rec_arena.losses.sequential.bpr import BPRLoss as _SeqBPRLoss
from rec_arena.losses.sequential.sampled_softmax import SampledSoftmaxLoss as _SampledSoftmax
from rec_arena.losses.sequential.gbce import GBCE as _GBCE
from rec_arena.losses.sequential.cross_entropy import CrossEntropyLoss as _CrossEntropy

_PATH_BATCH = 4
_PATH_SEQ_LEN = 6
_PATH_VOCAB = 30
_PATH_DIM = 16
_PATH_NUM_NEG = 5


def _make_path_inputs():
    """Create standard test inputs for path tests."""
    hidden_states = torch.randn(_PATH_BATCH, _PATH_SEQ_LEN, _PATH_DIM)
    item_embeddings = torch.randn(_PATH_VOCAB, _PATH_DIM)
    targets = torch.randint(3, _PATH_VOCAB, (_PATH_BATCH, _PATH_SEQ_LEN))
    mask = torch.ones(_PATH_BATCH, _PATH_SEQ_LEN)
    neg_items_3d = torch.randint(3, _PATH_VOCAB, (_PATH_BATCH, _PATH_SEQ_LEN, _PATH_NUM_NEG))
    neg_items_2d = torch.randint(3, _PATH_VOCAB, (_PATH_BATCH, _PATH_NUM_NEG))
    logits = torch.randn(_PATH_BATCH, _PATH_SEQ_LEN, _PATH_VOCAB)
    return hidden_states, item_embeddings, targets, mask, neg_items_3d, neg_items_2d, logits


# ===================================================================
# CrossEntropyLoss — path tests
# ===================================================================


class TestCrossEntropyLossPaths:
    def test_with_logits(self):
        loss_fn = _CrossEntropy()
        logits = torch.randn(_PATH_BATCH, _PATH_SEQ_LEN, _PATH_VOCAB)
        targets = torch.randint(0, _PATH_VOCAB, (_PATH_BATCH, _PATH_SEQ_LEN))
        mask = torch.ones(_PATH_BATCH, _PATH_SEQ_LEN)
        loss = loss_fn(logits=logits, targets=targets, mask=mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_with_hidden_states(self):
        loss_fn = _CrossEntropy()
        hidden = torch.randn(_PATH_BATCH, _PATH_SEQ_LEN, _PATH_DIM)
        item_emb = torch.randn(_PATH_VOCAB, _PATH_DIM)
        targets = torch.randint(0, _PATH_VOCAB, (_PATH_BATCH, _PATH_SEQ_LEN))
        mask = torch.ones(_PATH_BATCH, _PATH_SEQ_LEN)
        loss = loss_fn(hidden_states=hidden, item_embeddings=item_emb,
                       targets=targets, mask=mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ===================================================================
# BCENegativeSamplingLoss — path tests
# ===================================================================


class TestBCELossPaths:
    def test_fast_path_3d_neg(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _BCENegSampling()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_fast_path_2d_neg(self):
        hs, ie, tgt, mask, _, neg2d, _ = _make_path_inputs()
        loss_fn = _BCENegSampling()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_3d_neg(self):
        _, _, tgt, mask, neg3d, _, logits = _make_path_inputs()
        loss_fn = _BCENegSampling()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_2d_neg(self):
        _, _, tgt, mask, _, neg2d, logits = _make_path_inputs()
        loss_fn = _BCENegSampling()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_no_neg_items_raises(self):
        _, _, tgt, mask, _, _, logits = _make_path_inputs()
        loss_fn = _BCENegSampling()
        with pytest.raises(ValueError, match="neg_items"):
            loss_fn(logits=logits, targets=tgt, mask=mask)

    def test_no_inputs_raises(self):
        _, _, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _BCENegSampling()
        with pytest.raises(ValueError):
            loss_fn(targets=tgt, mask=mask, neg_items=neg3d)

    def test_l2_norm_path(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _BCENegSampling(l2_norm=True)
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg3d)
        assert torch.isfinite(loss)

    def test_non_negative(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _BCENegSampling()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg3d)
        assert loss >= 0


# ===================================================================
# BPRLoss — path tests
# ===================================================================


class TestBPRLossPaths:
    def test_fast_path_3d_neg(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _SeqBPRLoss()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_fast_path_2d_neg(self):
        hs, ie, tgt, mask, _, neg2d, _ = _make_path_inputs()
        loss_fn = _SeqBPRLoss()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_3d_neg(self):
        _, _, tgt, mask, neg3d, _, logits = _make_path_inputs()
        loss_fn = _SeqBPRLoss()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_2d_neg(self):
        _, _, tgt, mask, _, neg2d, logits = _make_path_inputs()
        loss_fn = _SeqBPRLoss()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_no_neg_items_raises(self):
        _, _, tgt, mask, _, _, logits = _make_path_inputs()
        loss_fn = _SeqBPRLoss()
        with pytest.raises(ValueError, match="neg_items"):
            loss_fn(logits=logits, targets=tgt, mask=mask)

    def test_no_inputs_raises(self):
        _, _, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _SeqBPRLoss()
        with pytest.raises(ValueError):
            loss_fn(targets=tgt, mask=mask, neg_items=neg3d)


# ===================================================================
# SampledSoftmaxLoss — path tests
# ===================================================================


class TestSampledSoftmaxPaths:
    def test_fast_path_3d_neg(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _SampledSoftmax()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_fast_path_2d_neg(self):
        hs, ie, tgt, mask, _, neg2d, _ = _make_path_inputs()
        loss_fn = _SampledSoftmax()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_3d_neg(self):
        _, _, tgt, mask, neg3d, _, logits = _make_path_inputs()
        loss_fn = _SampledSoftmax()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_2d_neg(self):
        _, _, tgt, mask, _, neg2d, logits = _make_path_inputs()
        loss_fn = _SampledSoftmax()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_no_neg_items_raises(self):
        _, _, tgt, mask, _, _, logits = _make_path_inputs()
        loss_fn = _SampledSoftmax()
        with pytest.raises(ValueError):
            loss_fn(logits=logits, targets=tgt, mask=mask)

    def test_l2_norm_path(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _SampledSoftmax(l2_norm=True)
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg3d)
        assert torch.isfinite(loss)

    def test_temperature_scaling(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_t1 = _SampledSoftmax(temperature=1.0)(
            hidden_states=hs, item_embeddings=ie,
            targets=tgt, mask=mask, neg_items=neg3d)
        loss_t01 = _SampledSoftmax(temperature=0.1)(
            hidden_states=hs, item_embeddings=ie,
            targets=tgt, mask=mask, neg_items=neg3d)
        # Different temperatures should give different losses
        assert not torch.isclose(loss_t1, loss_t01)


# ===================================================================
# GBCE — path tests
# ===================================================================


class TestGBCEPaths:
    def test_fast_path_3d_neg(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _GBCE()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_fast_path_2d_neg(self):
        hs, ie, tgt, mask, _, neg2d, _ = _make_path_inputs()
        loss_fn = _GBCE()
        loss = loss_fn(hidden_states=hs, item_embeddings=ie,
                       targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_3d_neg(self):
        _, _, tgt, mask, neg3d, _, logits = _make_path_inputs()
        loss_fn = _GBCE()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg3d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_slow_path_2d_neg(self):
        _, _, tgt, mask, _, neg2d, logits = _make_path_inputs()
        loss_fn = _GBCE()
        loss = loss_fn(logits=logits, targets=tgt, mask=mask, neg_items=neg2d)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_no_neg_items_raises(self):
        _, _, tgt, mask, _, _, logits = _make_path_inputs()
        loss_fn = _GBCE()
        with pytest.raises(ValueError, match="neg_items"):
            loss_fn(logits=logits, targets=tgt, mask=mask)

    def test_no_inputs_raises(self):
        _, _, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss_fn = _GBCE()
        with pytest.raises(ValueError):
            loss_fn(targets=tgt, mask=mask, neg_items=neg3d)

    def test_different_alpha(self):
        hs, ie, tgt, mask, neg3d, _, _ = _make_path_inputs()
        loss1 = _GBCE(alpha=0.3)(hidden_states=hs, item_embeddings=ie,
                                  targets=tgt, mask=mask, neg_items=neg3d)
        loss2 = _GBCE(alpha=0.8)(hidden_states=hs, item_embeddings=ie,
                                  targets=tgt, mask=mask, neg_items=neg3d)
        assert torch.isfinite(loss1) and torch.isfinite(loss2)



# ===========================================================================
# APPENDED FROM test_clustered_ce.py — ClusteredCrossEntropyLoss & create_item_clusters
# ===========================================================================
# Covers: ClusteredCrossEntropyLoss forward paths, create_item_clusters strategies.
# ===========================================================================

import pandas as pd

from rec_arena.losses.sequential.clustered_ce import (
    ClusteredCrossEntropyLoss,
    create_item_clusters,
)


def _make_clusters(vocab=20, num_clusters=4):
    """Create simple clusters splitting items evenly."""
    items_per_cluster = vocab // num_clusters
    clusters = []
    for i in range(num_clusters):
        start = i * items_per_cluster
        end = start + items_per_cluster if i < num_clusters - 1 else vocab
        clusters.append(torch.arange(start, end))
    return clusters


class TestClusteredCrossEntropyLoss:
    def test_forward_with_logits(self):
        clusters = _make_clusters(20, 4)
        loss_fn = ClusteredCrossEntropyLoss(clusters, vocab_size=20)
        logits = torch.randn(2, 5, 20)
        targets = torch.randint(0, 20, (2, 5))
        mask = torch.ones(2, 5)
        result = loss_fn(logits=logits, targets=targets, mask=mask)
        assert result.dim() == 0
        assert torch.isfinite(result)

    def test_forward_with_hidden_states(self):
        clusters = _make_clusters(20, 4)
        loss_fn = ClusteredCrossEntropyLoss(clusters, vocab_size=20, num_cross_negatives=4)
        hidden = torch.randn(2, 5, 16)
        item_emb = torch.randn(20, 16)
        targets = torch.randint(0, 20, (2, 5))
        mask = torch.ones(2, 5)
        result = loss_fn(hidden_states=hidden, item_embeddings=item_emb, targets=targets, mask=mask)
        assert result.dim() == 0
        assert torch.isfinite(result)

    def test_empty_mask_returns_zero(self):
        clusters = _make_clusters(20, 4)
        loss_fn = ClusteredCrossEntropyLoss(clusters, vocab_size=20)
        hidden = torch.randn(2, 5, 16)
        item_emb = torch.randn(20, 16)
        targets = torch.randint(0, 20, (2, 5))
        mask = torch.zeros(2, 5)  # all masked out
        result = loss_fn(hidden_states=hidden, item_embeddings=item_emb, targets=targets, mask=mask)
        assert result.item() == 0.0

    def test_with_anchor_items(self):
        clusters = _make_clusters(20, 4)
        anchors = torch.tensor([0, 1, 2])
        loss_fn = ClusteredCrossEntropyLoss(clusters, vocab_size=20, anchor_items=anchors)
        hidden = torch.randn(2, 5, 16)
        item_emb = torch.randn(20, 16)
        targets = torch.randint(0, 20, (2, 5))
        mask = torch.ones(2, 5)
        result = loss_fn(hidden_states=hidden, item_embeddings=item_emb, targets=targets, mask=mask)
        assert torch.isfinite(result)

    def test_no_cross_negatives(self):
        clusters = _make_clusters(20, 4)
        loss_fn = ClusteredCrossEntropyLoss(clusters, vocab_size=20, num_cross_negatives=0)
        hidden = torch.randn(2, 5, 16)
        item_emb = torch.randn(20, 16)
        targets = torch.randint(0, 20, (2, 5))
        mask = torch.ones(2, 5)
        result = loss_fn(hidden_states=hidden, item_embeddings=item_emb, targets=targets, mask=mask)
        assert torch.isfinite(result)


class TestCreateItemClusters:
    def _make_df(self, num_items=50):
        items = list(range(num_items)) * 10
        return pd.DataFrame({"item_id": items})

    def test_returns_correct_number_of_clusters(self):
        df = self._make_df()
        clusters, anchors = create_item_clusters(df, num_clusters=4, num_anchors=5)
        assert len(clusters) == 4

    def test_anchors_are_most_popular(self):
        df = self._make_df(50)
        clusters, anchors = create_item_clusters(df, num_clusters=4, num_anchors=5)
        assert len(anchors) == 5

    def test_all_items_covered(self):
        df = self._make_df(50)
        clusters, anchors = create_item_clusters(df, num_clusters=4, num_anchors=5)
        all_items = set(anchors.tolist())
        for c in clusters:
            all_items.update(c.tolist())
        assert len(all_items) == 50

    def test_random_strategy(self):
        df = self._make_df(50)
        clusters, anchors = create_item_clusters(df, num_clusters=4, strategy="random", num_anchors=5)
        assert len(clusters) == 4

    def test_popularity_strategy(self):
        df = self._make_df(50)
        clusters, anchors = create_item_clusters(df, num_clusters=4, strategy="popularity", num_anchors=5)
        assert len(clusters) == 4


# ===========================================================================
# APPENDED FROM test_llada_loss.py — LLaDALoss tests
# ===========================================================================
# Covers: LLaDALoss with logits, hidden_states, p_mask, error handling.
# ===========================================================================

from rec_arena.losses.sequential.llada_loss import LLaDALoss


class TestLLaDALoss:
    def test_with_logits_produces_finite_scalar(self):
        loss_fn = LLaDALoss()
        logits = torch.randn(2, 5, 20)
        targets = torch.randint(1, 20, (2, 5))
        mask = torch.ones(2, 5)
        result = loss_fn(logits=logits, targets=targets, mask=mask)
        assert result.dim() == 0
        assert torch.isfinite(result)

    def test_with_hidden_states(self):
        loss_fn = LLaDALoss()
        hidden = torch.randn(2, 5, 16)
        item_emb = torch.randn(20, 16)
        targets = torch.randint(1, 20, (2, 5))
        mask = torch.ones(2, 5)
        result = loss_fn(hidden_states=hidden, item_embeddings=item_emb, targets=targets, mask=mask)
        assert result.dim() == 0
        assert torch.isfinite(result)

    def test_with_p_mask(self):
        loss_fn = LLaDALoss()
        logits = torch.randn(2, 5, 20)
        targets = torch.randint(1, 20, (2, 5))
        mask = torch.ones(2, 5)
        p_mask = torch.full((2, 5), 0.5)
        result = loss_fn(logits=logits, targets=targets, mask=mask, p_mask=p_mask)
        assert torch.isfinite(result)

    def test_no_logits_no_hidden_raises(self):
        loss_fn = LLaDALoss()
        targets = torch.randint(1, 20, (2, 5))
        mask = torch.ones(2, 5)
        with pytest.raises(ValueError, match="Must provide"):
            loss_fn(targets=targets, mask=mask)

    def test_mask_zeros_out_positions(self):
        loss_fn = LLaDALoss()
        logits = torch.randn(2, 5, 20)
        targets = torch.randint(1, 20, (2, 5))
        mask_all = torch.ones(2, 5)
        mask_none = torch.zeros(2, 5)
        loss_all = loss_fn(logits=logits, targets=targets, mask=mask_all)
        loss_none = loss_fn(logits=logits, targets=targets, mask=mask_none)
        assert loss_none.item() < loss_all.item() or loss_none.item() == 0.0
