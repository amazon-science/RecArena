"""Shared fixtures and hypothesis strategies for the rec_arena test suite."""

import pandas as pd
import pytest
import torch
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_ITEMS = 20
NUM_USERS = 5

# ---------------------------------------------------------------------------
# Pytest Fixtures — Interaction DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_interactions_df() -> pd.DataFrame:
    """DataFrame with user_id, item_id, timestamp columns.

    5 users (0-4) × 10 items (1-10), timestamps monotonically increasing per user.
    Schema:
        user_id  : int in [0, num_users)
        item_id  : int in [1, num_items]  (1-indexed, 0 reserved for padding)
        timestamp: int, monotonically increasing per user
    """
    rows = []
    for user_id in range(NUM_USERS):
        for rank, item_id in enumerate(range(1, 11), start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": user_id * 100 + rank,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pytest Fixtures — Tensors
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_predictions() -> torch.Tensor:
    """Random score tensor of shape [BATCH_SIZE, NUM_ITEMS] on CPU."""
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, NUM_ITEMS)


@pytest.fixture
def synthetic_targets_1d() -> torch.Tensor:
    """1-D target tensor of shape [BATCH_SIZE] with valid item indices in [0, NUM_ITEMS)."""
    torch.manual_seed(42)
    return torch.randint(0, NUM_ITEMS, (BATCH_SIZE,))


@pytest.fixture
def synthetic_targets_2d() -> torch.Tensor:
    """2-D binary relevance tensor of shape [BATCH_SIZE, NUM_ITEMS]."""
    torch.manual_seed(42)
    targets = torch.zeros(BATCH_SIZE, NUM_ITEMS)
    for i in range(BATCH_SIZE):
        # Mark exactly one item as relevant per row
        targets[i, torch.randint(0, NUM_ITEMS, (1,)).item()] = 1.0
    return targets


# ---------------------------------------------------------------------------
# Pytest Fixtures — Config Dicts
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config_dict() -> dict:
    """Minimal valid base config dict with vocab_size and embedding_dim."""
    return {
        "vocab_size": 100,
        "embedding_dim": 64,
    }


@pytest.fixture
def sequential_config_dict() -> dict:
    """Valid sequential model config dict extending base fields."""
    return {
        "vocab_size": 100,
        "embedding_dim": 64,
        "max_seq_length": 50,
        "loss_type": "cross_entropy",
    }


@pytest.fixture
def implicit_config_dict() -> dict:
    """Valid implicit model config dict."""
    return {
        "num_users": 50,
        "num_items": 100,
        "embedding_dim": 64,
        "loss_type": "bce",
    }


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


def valid_predictions(batch: int = BATCH_SIZE, items: int = NUM_ITEMS) -> st.SearchStrategy:
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


def valid_k(max_items: int = NUM_ITEMS) -> st.SearchStrategy:
    """Strategy that generates an int in [1, max_items]."""
    return st.integers(min_value=1, max_value=max_items)


def valid_base_config() -> st.SearchStrategy:
    """Strategy that generates a dict with positive vocab_size and embedding_dim."""
    return st.fixed_dictionaries(
        {
            "vocab_size": st.integers(min_value=1, max_value=10_000),
            "embedding_dim": st.integers(min_value=1, max_value=512),
        }
    )


def whitespace_strings() -> st.SearchStrategy:
    """Strategy that generates strings composed entirely of whitespace characters."""
    whitespace_chars = " \t\n\r\x0b\x0c"
    return st.text(alphabet=whitespace_chars, min_size=1)


def valid_sequences(max_len: int = 10, vocab: int = 100) -> st.SearchStrategy:
    """Strategy that generates a list of ints in [1, vocab] with random length in [1, max_len]."""
    return st.integers(min_value=1, max_value=max_len).flatmap(
        lambda length: st.lists(
            st.integers(min_value=1, max_value=vocab),
            min_size=length,
            max_size=length,
        )
    )


# ---------------------------------------------------------------------------
# Hypothesis Strategies — Preprocessing Pipeline
# ---------------------------------------------------------------------------


@st.composite
def interactions_dataframes(
    draw: st.DrawFn,
    min_rows: int = 1,
    max_rows: int = 80,
    with_rating: bool | None = None,
    with_timestamp: bool = True,
    with_extras: bool = False,
) -> pd.DataFrame:
    """Generate valid interactions DataFrames with configurable columns.

    Args:
        min_rows: Minimum number of rows.
        max_rows: Maximum number of rows.
        with_rating: ``True`` always includes a ``rating`` column,
            ``False`` never includes it, ``None`` randomly decides.
        with_timestamp: Whether to include a ``timestamp`` column.
        with_extras: Whether to include extra columns (``extra_a``, ``extra_b``).
    """
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    user_ids = draw(
        st.lists(st.integers(min_value=0, max_value=10), min_size=n, max_size=n)
    )
    item_ids = draw(
        st.lists(st.integers(min_value=0, max_value=10), min_size=n, max_size=n)
    )
    data: dict = {"user_id": user_ids, "item_id": item_ids}

    if with_timestamp:
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

    if with_extras:
        data["extra_a"] = draw(
            st.lists(st.integers(min_value=0, max_value=100), min_size=n, max_size=n)
        )
        data["extra_b"] = draw(
            st.lists(st.integers(min_value=0, max_value=100), min_size=n, max_size=n)
        )

    return pd.DataFrame(data)


@st.composite
def preprocessing_pipelines(draw: st.DrawFn) -> "PreprocessingPipeline":
    """Generate random valid ``PreprocessingPipeline`` instances from the registry."""
    from rec_arena.datasets.preprocessing import (
        DuplicateInteractionRemover,
        ImplicitThresholdFilter,
        MinInteractionFilter,
        PreprocessingPipeline,
        TimestampNormalizer,
    )

    num_steps = draw(st.integers(min_value=0, max_value=4))
    steps = []
    for _ in range(num_steps):
        choice = draw(
            st.sampled_from([
                "min_interaction_filter",
                "implicit_threshold_filter",
                "timestamp_normalizer",
                "duplicate_interaction_remover",
            ])
        )
        if choice == "min_interaction_filter":
            steps.append(
                MinInteractionFilter(
                    min_interactions=draw(st.integers(min_value=1, max_value=5))
                )
            )
        elif choice == "implicit_threshold_filter":
            steps.append(
                ImplicitThresholdFilter(
                    threshold=draw(
                        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
                    )
                )
            )
        elif choice == "timestamp_normalizer":
            steps.append(TimestampNormalizer())
        else:
            steps.append(DuplicateInteractionRemover())
    return PreprocessingPipeline(steps)
