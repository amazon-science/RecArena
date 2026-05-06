"""Property-based and unit tests for the preprocessing pipeline.

Tests cover correctness properties defined in the design document for
MinInteractionFilter, ImplicitThresholdFilter, TimestampNormalizer,
and DuplicateInteractionRemover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from rec_arena.datasets.preprocessing import (
    DuplicateInteractionRemover,
    ImplicitThresholdFilter,
    MinInteractionFilter,
    PreprocessingPipeline,
    TimestampNormalizer,
    PREPROCESSOR_REGISTRY,
    create_default_pipeline,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def interactions_df(
    draw: st.DrawFn,
    min_rows: int = 1,
    max_rows: int = 80,
    with_rating: bool | None = None,
    with_timestamp: bool = True,
) -> pd.DataFrame:
    """Generate a valid interactions DataFrame."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    user_ids = draw(
        st.lists(st.integers(min_value=0, max_value=10), min_size=n, max_size=n)
    )
    item_ids = draw(
        st.lists(st.integers(min_value=0, max_value=10), min_size=n, max_size=n)
    )
    data: dict = {"user_id": user_ids, "item_id": item_ids}

    if with_timestamp or with_timestamp is None:
        timestamps = draw(
            st.lists(st.integers(min_value=0, max_value=10_000), min_size=n, max_size=n)
        )
        data["timestamp"] = timestamps

    include_rating = draw(st.booleans()) if with_rating is None else with_rating
    if include_rating:
        ratings = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        data["rating"] = ratings

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Property 3: MinInteractionFilter Rejects Invalid Parameters
# Feature: dataset-preprocessing, Property 3: MinInteractionFilter Rejects Invalid Parameters
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------


@given(value=st.integers(max_value=0))
@settings(max_examples=100)
def test_property_min_interaction_filter_rejects_invalid(value: int) -> None:
    """Any min_interactions < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="min_interactions must be >= 1"):
        MinInteractionFilter(min_interactions=value)


# ---------------------------------------------------------------------------
# Property 4: MinInteractionFilter Convergence Invariant
# Feature: dataset-preprocessing, Property 4: MinInteractionFilter Convergence Invariant
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=0, max_rows=80, with_timestamp=True),
    min_inter=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_property_min_interaction_filter_convergence(
    df: pd.DataFrame, min_inter: int
) -> None:
    """Every user and item in the output has >= min_interactions interactions (or output is empty)."""
    filt = MinInteractionFilter(min_interactions=min_inter)
    result = filt.transform(df)

    if result.empty:
        return

    user_counts = result["user_id"].value_counts()
    item_counts = result["item_id"].value_counts()
    assert (user_counts >= min_inter).all(), (
        f"Some users have fewer than {min_inter} interactions"
    )
    assert (item_counts >= min_inter).all(), (
        f"Some items have fewer than {min_inter} interactions"
    )


# ---------------------------------------------------------------------------
# Property 5: ImplicitThresholdFilter Rating Correctness
# Feature: dataset-preprocessing, Property 5: ImplicitThresholdFilter Rating Correctness
# Validates: Requirements 4.2, 4.3
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=1, max_rows=80, with_rating=True, with_timestamp=True),
    threshold=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_implicit_threshold_correctness(
    df: pd.DataFrame, threshold: float
) -> None:
    """Output contains only rows where rating >= threshold, all with implicit == 1."""
    filt = ImplicitThresholdFilter(threshold=threshold)
    result = filt.transform(df)

    if result.empty:
        return

    assert (result["rating"] >= threshold).all(), "Found rows with rating below threshold"
    assert (result["implicit"] == 1).all(), "Found rows with implicit != 1"


# ---------------------------------------------------------------------------
# Property 6: ImplicitThresholdFilter No-Rating Passthrough
# Feature: dataset-preprocessing, Property 6: ImplicitThresholdFilter No-Rating Passthrough
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=1, max_rows=80, with_rating=False, with_timestamp=True),
    threshold=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_implicit_threshold_no_rating(
    df: pd.DataFrame, threshold: float
) -> None:
    """Without a rating column, row count is unchanged and implicit == 1 everywhere."""
    assume("rating" not in df.columns)
    filt = ImplicitThresholdFilter(threshold=threshold)
    result = filt.transform(df)

    assert len(result) == len(df), "Row count changed without a rating column"
    assert (result["implicit"] == 1).all(), "Found rows with implicit != 1"


# ---------------------------------------------------------------------------
# Property 7: TimestampNormalizer Datetime Conversion
# Feature: dataset-preprocessing, Property 7: TimestampNormalizer Datetime Conversion
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=100)
def test_property_timestamp_normalizer_datetime(n: int) -> None:
    """Datetime timestamps are converted to integer Unix epoch seconds, preserving order."""
    # Build a DataFrame with datetime timestamps
    base = pd.Timestamp("2020-01-01")
    timestamps = [base + pd.Timedelta(seconds=i * 100) for i in range(n)]
    df = pd.DataFrame(
        {
            "user_id": list(range(n)),
            "item_id": list(range(n)),
            "timestamp": timestamps,
        }
    )

    normalizer = TimestampNormalizer()
    result = normalizer.transform(df)

    # Values should be integer Unix epoch seconds
    expected = [int(ts.timestamp()) for ts in timestamps]
    assert list(result["timestamp"]) == expected

    # Temporal ordering preserved
    ts_vals = result["timestamp"].tolist()
    assert ts_vals == sorted(ts_vals)


# ---------------------------------------------------------------------------
# Property 8: TimestampNormalizer Numeric Idempotence
# Feature: dataset-preprocessing, Property 8: TimestampNormalizer Numeric Idempotence
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=1, max_rows=80, with_timestamp=True),
)
@settings(max_examples=100)
def test_property_timestamp_normalizer_numeric(df: pd.DataFrame) -> None:
    """Numeric timestamps are left unchanged."""
    normalizer = TimestampNormalizer()
    original_ts = df["timestamp"].tolist()
    result = normalizer.transform(df)
    assert result["timestamp"].tolist() == original_ts


# ---------------------------------------------------------------------------
# Property 9: TimestampNormalizer Missing Column
# Feature: dataset-preprocessing, Property 9: TimestampNormalizer Missing Column
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=80),
)
@settings(max_examples=100)
def test_property_timestamp_normalizer_missing(n: int) -> None:
    """Missing timestamp column gets sequential integers [0, 1, ..., n-1]."""
    df = pd.DataFrame(
        {
            "user_id": list(range(n)),
            "item_id": list(range(n)),
        }
    )
    normalizer = TimestampNormalizer()
    result = normalizer.transform(df)

    assert "timestamp" in result.columns
    assert list(result["timestamp"]) == list(range(n))


# ---------------------------------------------------------------------------
# Property 10: DuplicateInteractionRemover Correctness
# Feature: dataset-preprocessing, Property 10: DuplicateInteractionRemover Correctness
# Validates: Requirements 6.1, 6.3
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=1, max_rows=80, with_timestamp=True),
)
@settings(max_examples=100)
def test_property_duplicate_remover_correctness(df: pd.DataFrame) -> None:
    """Output row count equals unique (user_id, item_id) pairs; kept row has max timestamp."""
    remover = DuplicateInteractionRemover()
    result = remover.transform(df)

    n_unique_pairs = df.groupby(["user_id", "item_id"]).ngroups
    assert len(result) == n_unique_pairs, (
        f"Expected {n_unique_pairs} rows, got {len(result)}"
    )

    # For each pair, the kept timestamp should be the max from the input
    expected_max = df.groupby(["user_id", "item_id"])["timestamp"].max()
    for _, row in result.iterrows():
        key = (row["user_id"], row["item_id"])
        assert row["timestamp"] == expected_max.loc[key], (
            f"Pair {key}: expected timestamp {expected_max.loc[key]}, got {row['timestamp']}"
        )


# ---------------------------------------------------------------------------
# Helpers for pipeline-level property tests
# ---------------------------------------------------------------------------


@st.composite
def interactions_df_with_extras(
    draw: st.DrawFn,
    min_rows: int = 1,
    max_rows: int = 50,
) -> pd.DataFrame:
    """Generate an interactions DataFrame with extra columns beyond the core set."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    user_ids = draw(
        st.lists(st.integers(min_value=0, max_value=10), min_size=n, max_size=n)
    )
    item_ids = draw(
        st.lists(st.integers(min_value=0, max_value=10), min_size=n, max_size=n)
    )
    timestamps = draw(
        st.lists(st.integers(min_value=0, max_value=10_000), min_size=n, max_size=n)
    )
    extra_a = draw(
        st.lists(st.integers(min_value=0, max_value=100), min_size=n, max_size=n)
    )
    extra_b = draw(
        st.lists(st.integers(min_value=0, max_value=100), min_size=n, max_size=n)
    )
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "timestamp": timestamps,
            "extra_a": extra_a,
            "extra_b": extra_b,
        }
    )


@st.composite
def random_preprocessor(draw: st.DrawFn) -> "Preprocessor":
    """Pick a random preprocessor with valid params from the registry."""
    choice = draw(st.sampled_from([
        "min_interaction_filter",
        "implicit_threshold_filter",
        "timestamp_normalizer",
        "duplicate_interaction_remover",
    ]))
    if choice == "min_interaction_filter":
        return MinInteractionFilter(
            min_interactions=draw(st.integers(min_value=1, max_value=5))
        )
    elif choice == "implicit_threshold_filter":
        return ImplicitThresholdFilter(
            threshold=draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False))
        )
    elif choice == "timestamp_normalizer":
        return TimestampNormalizer()
    else:
        return DuplicateInteractionRemover()


# ---------------------------------------------------------------------------
# Property 1: Column Preservation
# Feature: dataset-preprocessing, Property 1: Column Preservation
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------


@given(
    df=interactions_df_with_extras(min_rows=1, max_rows=50),
    preprocessor=random_preprocessor(),
)
@settings(max_examples=100)
def test_property_column_preservation(
    df: pd.DataFrame, preprocessor
) -> None:
    """Any preprocessor preserves all columns not explicitly modified."""
    input_cols = set(df.columns)
    result = preprocessor.transform(df)
    output_cols = set(result.columns)
    # All input columns must still be present (preprocessors may add columns but not drop)
    assert input_cols.issubset(output_cols), (
        f"Columns lost: {input_cols - output_cols} by {preprocessor.name}"
    )


# ---------------------------------------------------------------------------
# Property 2: Pipeline Sequential Composition
# Feature: dataset-preprocessing, Property 2: Pipeline Sequential Composition
# Validates: Requirements 2.2, 2.4
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=0, max_rows=50, with_timestamp=True),
    steps=st.lists(random_preprocessor(), min_size=0, max_size=4),
)
@settings(max_examples=100)
def test_property_pipeline_sequential_composition(
    df: pd.DataFrame, steps: list
) -> None:
    """Pipeline.transform(df) equals manually chaining each step."""
    pipeline = PreprocessingPipeline(steps)
    pipeline_result = pipeline.transform(df)

    # Manual sequential application (with same short-circuit on empty)
    manual = df
    for step in steps:
        manual = step.transform(manual)
        if manual.empty:
            break

    pd.testing.assert_frame_equal(
        pipeline_result.reset_index(drop=True),
        manual.reset_index(drop=True),
    )



# ---------------------------------------------------------------------------
# Helper: replicate BaseDataset.load_data() inline preprocessing
# ---------------------------------------------------------------------------


def _inline_preprocess(
    df: pd.DataFrame,
    min_interactions: int,
    implicit_threshold: float | None,
) -> pd.DataFrame:
    """Reproduce the exact inline logic from BaseDataset.load_data()."""
    df = df.copy()
    if implicit_threshold is not None and "rating" in df.columns:
        if "implicit" not in df.columns:
            df["implicit"] = (df["rating"] >= implicit_threshold).astype(int)
        df = df[df["implicit"] == 1].copy()
    elif "implicit" not in df.columns:
        df["implicit"] = 1

    # _filter_by_min_interactions
    if min_interactions <= 1:
        return df
    prev_size = len(df) + 1
    while len(df) < prev_size:
        prev_size = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[
            df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)
        ].copy()
    return df


# ---------------------------------------------------------------------------
# Property 11: Default Pipeline Equivalence
# Feature: dataset-preprocessing, Property 11: Default Pipeline Equivalence
# Validates: Requirements 8.2
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=0, max_rows=80, with_rating=True, with_timestamp=True),
    min_inter=st.integers(min_value=1, max_value=5),
    threshold=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=100)
def test_property_default_pipeline_equivalence(
    df: pd.DataFrame, min_inter: int, threshold: float | None
) -> None:
    """create_default_pipeline output matches the inline BaseDataset logic."""
    pipeline = create_default_pipeline(
        min_interactions=min_inter, implicit_threshold=threshold
    )
    pipeline_result = pipeline.transform(df).reset_index(drop=True)
    inline_result = _inline_preprocess(df, min_inter, threshold).reset_index(drop=True)

    # Sort both by the same columns for stable comparison
    sort_cols = ["user_id", "item_id", "timestamp"]
    pipeline_sorted = pipeline_result.sort_values(sort_cols).reset_index(drop=True)
    inline_sorted = inline_result.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(pipeline_sorted, inline_sorted)


# ---------------------------------------------------------------------------
# Property 12: Pipeline Serialization Round-Trip
# Feature: dataset-preprocessing, Property 12: Pipeline Serialization Round-Trip
# Validates: Requirements 9.3
# ---------------------------------------------------------------------------


@given(
    df=interactions_df(min_rows=0, max_rows=50, with_timestamp=True),
    steps=st.lists(random_preprocessor(), min_size=0, max_size=4),
)
@settings(max_examples=100)
def test_property_pipeline_serialization_roundtrip(
    df: pd.DataFrame, steps: list
) -> None:
    """from_dict(pipeline.to_dict()).transform(df) equals pipeline.transform(df)."""
    pipeline = PreprocessingPipeline(steps)
    restored = PreprocessingPipeline.from_dict(pipeline.to_dict())

    original_result = pipeline.transform(df).reset_index(drop=True)
    restored_result = restored.transform(df).reset_index(drop=True)

    pd.testing.assert_frame_equal(original_result, restored_result)


# ---------------------------------------------------------------------------
# BaseDataset Integration Tests
# Requirements: 7.1, 7.2, 7.3, 7.4
# ---------------------------------------------------------------------------

from rec_arena.datasets.base_dataset import BaseDataset


class _StubDataset(BaseDataset):
    """Minimal concrete BaseDataset backed by an in-memory DataFrame."""

    def __init__(self, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self._raw_df = df

    def _get_default_split_type(self) -> str:
        return "leave_one_out"

    def _load_raw_data(self) -> pd.DataFrame:
        return self._raw_df.copy()


def _make_integration_df(num_users: int = 5, items_per_user: int = 10) -> pd.DataFrame:
    """Create a simple interactions DataFrame for integration tests."""
    rows = []
    for u in range(num_users):
        for rank, i in enumerate(range(items_per_user)):
            rows.append({
                "user_id": u + 100,
                "item_id": i + 200,
                "timestamp": u * 100 + rank,
            })
    return pd.DataFrame(rows)


class TestBaseDatasetPipelineIntegration:
    """Tests for BaseDataset integration with PreprocessingPipeline."""

    def test_pipeline_called_between_load_and_remap(self):
        """Pipeline transform is applied between _load_raw_data and _remap_ids."""
        df = _make_integration_df(5, 10)
        # Pipeline that adds an implicit column — proves it ran before remap
        pipeline = PreprocessingPipeline([
            ImplicitThresholdFilter(threshold=None),
        ])
        ds = _StubDataset(df, min_interactions=1, preprocessing_pipeline=pipeline)
        ds.load_data()

        assert "implicit" in ds.interactions_df.columns
        assert (ds.interactions_df["implicit"] == 1).all()
        # Remap should have happened: user_ids start at 0, item_ids start at 1
        assert ds.interactions_df["user_id"].min() == 0
        assert ds.interactions_df["item_id"].min() == 1

    def test_backward_compatibility_no_pipeline(self):
        """When no pipeline is provided, existing inline logic runs unchanged."""
        df = _make_integration_df(5, 10)
        df["rating"] = [5.0, 4.0, 3.0, 2.0, 1.0] * 10
        ds = _StubDataset(df, min_interactions=1, implicit_threshold=4.0)
        ds.load_data()

        # Inline logic should have filtered by rating >= 4.0
        assert len(ds.interactions_df) < 50
        assert "implicit" in ds.interactions_df.columns
        assert (ds.interactions_df["implicit"] == 1).all()

    def test_pipeline_empty_output_raises_valueerror(self):
        """ValueError raised when pipeline produces an empty DataFrame."""
        df = _make_integration_df(3, 5)
        df["rating"] = 1.0  # All ratings below threshold
        # Pipeline that filters everything out
        pipeline = PreprocessingPipeline([
            ImplicitThresholdFilter(threshold=5.0),
        ])
        ds = _StubDataset(df, min_interactions=1, preprocessing_pipeline=pipeline)

        with pytest.raises(ValueError, match="Preprocessing pipeline produced an empty DataFrame"):
            ds.load_data()

    def test_pipeline_with_min_interaction_filter(self):
        """Pipeline with MinInteractionFilter works end-to-end."""
        df = _make_integration_df(5, 10)
        pipeline = PreprocessingPipeline([
            ImplicitThresholdFilter(threshold=None),
            MinInteractionFilter(min_interactions=1),
        ])
        ds = _StubDataset(df, min_interactions=1, preprocessing_pipeline=pipeline)
        ds.load_data()

        assert ds.num_users == 5
        assert ds.num_items == 10
        assert len(ds.interactions_df) == 50

    def test_pipeline_replaces_inline_logic(self):
        """When pipeline is provided, inline implicit_threshold logic is skipped."""
        df = _make_integration_df(3, 5)
        df["rating"] = 5.0  # All high ratings
        # Pipeline does NOT filter by rating — just adds implicit=1
        pipeline = PreprocessingPipeline([
            ImplicitThresholdFilter(threshold=None),
        ])
        # Even though implicit_threshold is set, pipeline takes precedence
        ds = _StubDataset(
            df,
            min_interactions=1,
            implicit_threshold=100.0,  # Would filter everything in inline mode
            preprocessing_pipeline=pipeline,
        )
        ds.load_data()

        # Pipeline ran (no filtering), so all 15 rows should remain
        assert len(ds.interactions_df) == 15


# ---------------------------------------------------------------------------
# Edge-case unit tests for uncovered lines
# ---------------------------------------------------------------------------


def test_duplicate_remover_empty_dataframe():
    """DuplicateInteractionRemover returns empty df unchanged (line 151)."""
    remover = DuplicateInteractionRemover()
    empty_df = pd.DataFrame(columns=["user_id", "item_id", "timestamp"])
    result = remover.transform(empty_df)
    assert result.empty
    assert list(result.columns) == ["user_id", "item_id", "timestamp"]


def test_from_dict_unknown_preprocessor_raises():
    """from_dict raises ValueError for unknown preprocessor name (lines 224-225)."""
    bad_config = {"steps": [{"name": "nonexistent_preprocessor", "params": {}}]}
    with pytest.raises(ValueError, match="Unknown preprocessor name"):
        PreprocessingPipeline.from_dict(bad_config)
