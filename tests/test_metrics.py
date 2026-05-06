"""Tests for rec_arena.metrics — unit tests and property-based tests.

Covers:
  - Individual metric functions: ndcg_at_k, recall_at_k, precision_at_k, hit_rate_at_k, mrr_at_k
  - MetricCalculator orchestrator class
  - Property-based tests for universal invariants
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rec_arena.metrics import (
    MetricCalculator,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Constants & strategies (mirroring conftest values)
# ---------------------------------------------------------------------------

NUM_ITEMS = 20
BATCH_SIZE = 4

ALL_METRIC_FNS = [ndcg_at_k, recall_at_k, precision_at_k, hit_rate_at_k, mrr_at_k]


def _valid_predictions(batch: int = BATCH_SIZE, items: int = NUM_ITEMS) -> st.SearchStrategy:
    """Strategy that generates a torch.Tensor of shape [batch, items] with finite floats."""
    return st.lists(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=items,
            max_size=items,
        ),
        min_size=batch,
        max_size=batch,
    ).map(lambda rows: torch.tensor(rows, dtype=torch.float32))


def _valid_k(max_items: int = NUM_ITEMS) -> st.SearchStrategy:
    """Strategy that generates an int in [1, max_items]."""
    return st.integers(min_value=1, max_value=max_items)


def _perfect_prediction_2d(num_items: int = 10, batch: int = 1) -> tuple:
    """Return (predictions, targets_2d) where the top-scored item is the only relevant one."""
    predictions = torch.zeros(batch, num_items)
    targets = torch.zeros(batch, num_items)
    for i in range(batch):
        predictions[i, 0] = 10.0  # item 0 gets highest score
        targets[i, 0] = 1.0  # item 0 is relevant
    return predictions, targets


# ============================================================================
# Task 2.1 — Unit tests for individual metric functions
# ============================================================================


class TestNdcgAtK:
    """Unit tests for ndcg_at_k — Requirements 1.1, 1.2, 1.3, 1.8, 1.9."""

    def test_range_with_2d_targets(self, synthetic_predictions, synthetic_targets_2d):
        """NDCG should be in [0, 1] for 2D relevance targets."""
        score = ndcg_at_k(synthetic_predictions, synthetic_targets_2d, k=5)
        assert 0.0 <= score <= 1.0

    def test_range_with_1d_targets(self, synthetic_predictions, synthetic_targets_1d):
        """NDCG should be in [0, 1] for 1D item-index targets."""
        score = ndcg_at_k(synthetic_predictions, synthetic_targets_1d, k=5)
        assert 0.0 <= score <= 1.0

    def test_perfect_prediction_returns_one(self):
        """When the target item is ranked first, NDCG should be 1.0."""
        preds, targets = _perfect_prediction_2d()
        assert ndcg_at_k(preds, targets, k=5) == pytest.approx(1.0)

    def test_empty_predictions_returns_zero(self):
        """Empty predictions tensor should yield 0.0."""
        empty = torch.tensor([])
        targets = torch.tensor([])
        assert ndcg_at_k(empty, targets, k=5) == 0.0

    def test_k_exceeds_num_items(self, synthetic_predictions, synthetic_targets_2d):
        """k larger than num_items should not raise and should return valid score."""
        score = ndcg_at_k(synthetic_predictions, synthetic_targets_2d, k=NUM_ITEMS + 100)
        assert 0.0 <= score <= 1.0

    def test_single_item_prediction(self):
        """1-D prediction (single sample, no batch dim) should work."""
        preds = torch.tensor([3.0, 1.0, 2.0])
        target = torch.tensor(0)  # item 0 has highest score
        score = ndcg_at_k(preds, target, k=3)
        assert 0.0 <= score <= 1.0


class TestRecallAtK:
    """Unit tests for recall_at_k — Requirements 1.4, 1.8, 1.9."""

    def test_range_with_2d_targets(self, synthetic_predictions, synthetic_targets_2d):
        score = recall_at_k(synthetic_predictions, synthetic_targets_2d, k=5)
        assert 0.0 <= score <= 1.0

    def test_range_with_1d_targets(self, synthetic_predictions, synthetic_targets_1d):
        score = recall_at_k(synthetic_predictions, synthetic_targets_1d, k=5)
        assert 0.0 <= score <= 1.0

    def test_empty_predictions_returns_zero(self):
        assert recall_at_k(torch.tensor([]), torch.tensor([]), k=5) == 0.0

    def test_perfect_prediction_2d(self):
        preds, targets = _perfect_prediction_2d()
        assert recall_at_k(preds, targets, k=5) == pytest.approx(1.0)

    def test_k_exceeds_num_items(self, synthetic_predictions, synthetic_targets_2d):
        score = recall_at_k(synthetic_predictions, synthetic_targets_2d, k=NUM_ITEMS + 100)
        assert 0.0 <= score <= 1.0


class TestPrecisionAtK:
    """Unit tests for precision_at_k — Requirements 1.5, 1.8, 1.9."""

    def test_range_with_2d_targets(self, synthetic_predictions, synthetic_targets_2d):
        score = precision_at_k(synthetic_predictions, synthetic_targets_2d, k=5)
        assert 0.0 <= score <= 1.0

    def test_range_with_1d_targets(self, synthetic_predictions, synthetic_targets_1d):
        score = precision_at_k(synthetic_predictions, synthetic_targets_1d, k=5)
        assert 0.0 <= score <= 1.0

    def test_empty_predictions_returns_zero(self):
        assert precision_at_k(torch.tensor([]), torch.tensor([]), k=5) == 0.0

    def test_perfect_prediction_2d(self):
        preds, targets = _perfect_prediction_2d()
        # 1 relevant item in top-5 → precision = 1/5 = 0.2
        score = precision_at_k(preds, targets, k=5)
        assert score == pytest.approx(0.2)

    def test_k_exceeds_num_items(self, synthetic_predictions, synthetic_targets_2d):
        score = precision_at_k(synthetic_predictions, synthetic_targets_2d, k=NUM_ITEMS + 100)
        assert 0.0 <= score <= 1.0

    def test_k_zero_returns_zero(self):
        """k=0 is an edge case that should return 0.0."""
        preds = torch.randn(2, 10)
        targets = torch.zeros(2, 10)
        assert precision_at_k(preds, targets, k=0) == 0.0


class TestHitRateAtK:
    """Unit tests for hit_rate_at_k — Requirements 1.6, 1.8, 1.9."""

    def test_range_with_2d_targets(self, synthetic_predictions, synthetic_targets_2d):
        score = hit_rate_at_k(synthetic_predictions, synthetic_targets_2d, k=5)
        assert 0.0 <= score <= 1.0

    def test_range_with_1d_targets(self, synthetic_predictions, synthetic_targets_1d):
        score = hit_rate_at_k(synthetic_predictions, synthetic_targets_1d, k=5)
        assert 0.0 <= score <= 1.0

    def test_empty_predictions_returns_zero(self):
        assert hit_rate_at_k(torch.tensor([]), torch.tensor([]), k=5) == 0.0

    def test_perfect_prediction_2d(self):
        preds, targets = _perfect_prediction_2d()
        assert hit_rate_at_k(preds, targets, k=5) == pytest.approx(1.0)

    def test_k_exceeds_num_items(self, synthetic_predictions, synthetic_targets_2d):
        score = hit_rate_at_k(synthetic_predictions, synthetic_targets_2d, k=NUM_ITEMS + 100)
        assert 0.0 <= score <= 1.0


class TestMrrAtK:
    """Unit tests for mrr_at_k — Requirements 1.7, 1.8, 1.9."""

    def test_range_with_2d_targets(self, synthetic_predictions, synthetic_targets_2d):
        score = mrr_at_k(synthetic_predictions, synthetic_targets_2d, k=5)
        assert 0.0 <= score <= 1.0

    def test_range_with_1d_targets(self, synthetic_predictions, synthetic_targets_1d):
        score = mrr_at_k(synthetic_predictions, synthetic_targets_1d, k=5)
        assert 0.0 <= score <= 1.0

    def test_empty_predictions_returns_zero(self):
        assert mrr_at_k(torch.tensor([]), torch.tensor([]), k=5) == 0.0

    def test_perfect_prediction_2d(self):
        preds, targets = _perfect_prediction_2d()
        # Relevant item at rank 1 → MRR = 1.0
        assert mrr_at_k(preds, targets, k=5) == pytest.approx(1.0)

    def test_k_exceeds_num_items(self, synthetic_predictions, synthetic_targets_2d):
        score = mrr_at_k(synthetic_predictions, synthetic_targets_2d, k=NUM_ITEMS + 100)
        assert 0.0 <= score <= 1.0


# ============================================================================
# Task 2.2 — Property test: metric range [0, 1]
# ============================================================================


# Feature: comprehensive-test-suite, Property 1: Metric functions return values in [0, 1]
class TestPropertyMetricRange:
    """Property 1: For any valid predictions and targets, metrics return values in [0, 1].

    Validates: Requirements 1.1, 1.4, 1.5, 1.6, 1.7, 1.8
    """

    @given(preds=_valid_predictions(), k=_valid_k())
    @settings(max_examples=100)
    def test_all_metrics_in_unit_range_2d(self, preds: torch.Tensor, k: int):
        """All metrics should return [0, 1] for 2D relevance targets."""
        batch = preds.size(0)
        num_items = preds.size(1)
        # Build random binary relevance with at least one relevant item per row
        targets = torch.zeros(batch, num_items)
        for i in range(batch):
            targets[i, torch.randint(0, num_items, (1,)).item()] = 1.0

        for fn in ALL_METRIC_FNS:
            score = fn(preds, targets, k)
            assert 0.0 <= score <= 1.0, f"{fn.__name__} returned {score}"

    @given(preds=_valid_predictions(), k=_valid_k())
    @settings(max_examples=100)
    def test_all_metrics_in_unit_range_1d(self, preds: torch.Tensor, k: int):
        """All metrics should return [0, 1] for 1D single-item targets."""
        batch = preds.size(0)
        num_items = preds.size(1)
        targets = torch.randint(0, num_items, (batch,))

        for fn in ALL_METRIC_FNS:
            score = fn(preds, targets, k)
            assert 0.0 <= score <= 1.0, f"{fn.__name__} returned {score}"


# ============================================================================
# Task 2.3 — MetricCalculator unit tests
# ============================================================================


class TestMetricCalculator:
    """Unit tests for MetricCalculator — Requirements 2.1–2.6."""

    def test_default_k_values(self):
        """Default k_values should be [5, 10, 20]."""
        calc = MetricCalculator()
        assert calc.k_values == [5, 10, 20]

    def test_calculate_all_returns_complete_keys(
        self, synthetic_predictions, synthetic_targets_2d
    ):
        """calculate_all should return keys for every metric × k combination."""
        calc = MetricCalculator()
        results = calc.calculate_all(synthetic_predictions, synthetic_targets_2d)

        expected_metrics = ["ndcg", "recall", "precision", "hit_rate", "mrr"]
        expected_ks = [5, 10, 20]
        for m in expected_metrics:
            for k in expected_ks:
                key = f"{m}@{k}"
                assert key in results, f"Missing key: {key}"

    def test_add_metric_callable(self, synthetic_predictions, synthetic_targets_2d):
        """add_metric with a callable should appear in subsequent results."""
        calc = MetricCalculator(k_values=[5])

        def dummy_metric(preds, targets, k):
            return 0.42

        calc.add_metric("dummy", dummy_metric)
        results = calc.calculate_all(synthetic_predictions, synthetic_targets_2d)
        assert "dummy@5" in results
        assert results["dummy@5"] == pytest.approx(0.42)

    def test_add_metric_non_callable_raises(self):
        """add_metric with a non-callable should raise ValueError."""
        calc = MetricCalculator()
        with pytest.raises(ValueError, match="callable"):
            calc.add_metric("bad", "not_a_function")

    def test_set_k_values_valid(self, synthetic_predictions, synthetic_targets_2d):
        """set_k_values with valid positive ints should update subsequent calculations."""
        calc = MetricCalculator()
        calc.set_k_values([3, 7])
        results = calc.calculate_all(synthetic_predictions, synthetic_targets_2d)
        # Should have keys for k=3 and k=7, not the old defaults
        assert "ndcg@3" in results
        assert "ndcg@7" in results
        assert "ndcg@5" not in results

    def test_set_k_values_non_positive_raises(self):
        """set_k_values with non-positive values should raise ValueError."""
        calc = MetricCalculator()
        with pytest.raises(ValueError):
            calc.set_k_values([0, 5])

    def test_set_k_values_non_integer_raises(self):
        """set_k_values with non-integer values should raise ValueError."""
        calc = MetricCalculator()
        with pytest.raises(ValueError):
            calc.set_k_values([1.5, 5])

    def test_set_k_values_negative_raises(self):
        """set_k_values with negative values should raise ValueError."""
        calc = MetricCalculator()
        with pytest.raises(ValueError):
            calc.set_k_values([-1, 10])


# ============================================================================
# Task 2.4 — Property test: MetricCalculator complete result keys
# ============================================================================


# Feature: comprehensive-test-suite, Property 2: MetricCalculator produces complete result keys
class TestPropertyMetricCalculatorKeys:
    """Property 2: calculate_all returns a key for every metric × k combination.

    Validates: Requirements 2.2, 2.5
    """

    @given(
        preds=_valid_predictions(),
        k_values=st.lists(
            st.integers(min_value=1, max_value=NUM_ITEMS), min_size=1, max_size=5, unique=True
        ),
    )
    @settings(max_examples=100)
    def test_complete_keys(self, preds: torch.Tensor, k_values: list):
        batch = preds.size(0)
        num_items = preds.size(1)
        targets = torch.zeros(batch, num_items)
        for i in range(batch):
            targets[i, torch.randint(0, num_items, (1,)).item()] = 1.0

        calc = MetricCalculator(k_values=k_values)
        results = calc.calculate_all(preds, targets)

        for metric_name in calc.metrics:
            for k in k_values:
                key = f"{metric_name}@{k}"
                assert key in results, f"Missing key: {key}"


# ============================================================================
# Task 2.5 — Property test: invalid k-values rejected
# ============================================================================


# Feature: comprehensive-test-suite, Property 3: Invalid k-values are rejected
class TestPropertyInvalidKValues:
    """Property 3: set_k_values rejects non-positive or non-integer values.

    Validates: Requirements 2.6
    """

    @given(bad_k=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_non_positive_k_rejected(self, bad_k: int):
        calc = MetricCalculator()
        with pytest.raises(ValueError):
            calc.set_k_values([bad_k])

    @given(bad_k=st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: not x.is_integer()))
    @settings(max_examples=100)
    def test_non_integer_k_rejected(self, bad_k: float):
        calc = MetricCalculator()
        with pytest.raises(ValueError):
            calc.set_k_values([bad_k])
