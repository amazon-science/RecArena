"""Comprehensive tests for rec_arena.datasets module.

Covers: split strategies, sequential/implicit datasets, samplers,
collate functions, augmentation, and caching.
"""

import random
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from rec_arena.datasets.split_strategies import (
    LeaveOneOutSplit,
    TemporalSplit,
    UserBasedSplit,
    get_split_strategy,
    SPLIT_STRATEGIES,
)
from rec_arena.datasets.sequential_dataset import (
    SequentialDataset,
    prepare_sequences,
    build_user_histories,
)
from rec_arena.datasets.implicit_dataset import (
    ImplicitDataset,
    prepare_implicit_interactions,
)
from rec_arena.datasets.samplers import UniformSampler, PopularitySampler
from rec_arena.datasets.collate import (
    SequentialNegativeSamplingCollate,
    ImplicitNegativeSamplingCollate,
    BatchSharedNegativeSamplingCollate,
    GraphNegativeSamplingCollate,
)
from rec_arena.datasets.augmentation import SequenceAugmenter
from rec_arena.datasets.cache import DatasetCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_USERS = 5
NUM_ITEMS = 20


def _make_interactions_df(num_users=NUM_USERS, items_per_user=10):
    """Build a synthetic interactions DataFrame."""
    rows = []
    for uid in range(num_users):
        for rank, iid in enumerate(range(1, items_per_user + 1), start=1):
            rows.append({"user_id": uid, "item_id": iid, "timestamp": uid * 100 + rank})
    return pd.DataFrame(rows)


def _make_sequential_batch(batch_size=4, seq_len=10, num_items=20):
    """Build a list of sequential sample dicts suitable for collate functions."""
    batch = []
    for i in range(batch_size):
        seq = list(range(3, 3 + seq_len))  # items starting at offset 3
        batch.append({
            "user_id": torch.tensor(i, dtype=torch.long),
            "sequence": torch.tensor(seq, dtype=torch.long),
            "sequence_length": torch.tensor(seq_len, dtype=torch.long),
        })
    return batch


def _make_implicit_batch(batch_size=4, item_offset=3):
    """Build a list of implicit sample dicts suitable for collate functions."""
    batch = []
    for i in range(batch_size):
        batch.append({
            "user_id": torch.tensor(i, dtype=torch.long),
            "item_id": torch.tensor(item_offset + i, dtype=torch.long),
        })
    return batch


def _make_graph_batch(batch_size=4, num_items=20, item_offset=3):
    """Build a list of graph sample dicts suitable for GraphNegativeSamplingCollate."""
    edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    batch = []
    for i in range(batch_size):
        batch.append({
            "user_id": torch.tensor(i, dtype=torch.long),
            "item_id": torch.tensor(item_offset + i, dtype=torch.long),
            "label": torch.tensor(1.0),
            "edge_index": edge_index,
        })
    return batch


# ===================================================================
# 5.1  Split strategy unit tests
# ===================================================================


class TestLeaveOneOutSplit:
    """Unit tests for LeaveOneOutSplit."""

    def test_users_with_3plus_interactions(self, synthetic_interactions_df):
        """Users with 3+ interactions: last→test, second-to-last→val, rest→train."""
        splitter = LeaveOneOutSplit(min_sequence_length=3)
        train, val, test = splitter.split(synthetic_interactions_df, NUM_USERS, NUM_ITEMS)

        for uid in range(NUM_USERS):
            user_rows = synthetic_interactions_df[
                synthetic_interactions_df["user_id"] == uid
            ].sort_values("timestamp")
            items = user_rows["item_id"].tolist()

            test_items = test[test["user_id"] == uid]["item_id"].tolist()
            val_items = val[val["user_id"] == uid]["item_id"].tolist()
            train_items = train[train["user_id"] == uid]["item_id"].tolist()

            assert test_items == [items[-1]]
            assert val_items == [items[-2]]
            assert train_items == items[:-2]

    def test_user_with_2_interactions(self):
        """User with exactly 2 interactions: last→test, first→train, no val."""
        df = pd.DataFrame([
            {"user_id": 0, "item_id": 1, "timestamp": 1},
            {"user_id": 0, "item_id": 2, "timestamp": 2},
        ])
        splitter = LeaveOneOutSplit(min_sequence_length=3)
        train, val, test = splitter.split(df, 1, 2)

        assert len(test) == 1
        assert test.iloc[0]["item_id"] == 2
        assert len(train) == 1
        assert train.iloc[0]["item_id"] == 1
        assert len(val) == 0

    def test_user_with_1_interaction(self):
        """User with exactly 1 interaction: goes to train only."""
        df = pd.DataFrame([{"user_id": 0, "item_id": 1, "timestamp": 1}])
        splitter = LeaveOneOutSplit(min_sequence_length=3)
        train, val, test = splitter.split(df, 1, 1)

        assert len(train) == 1
        assert len(val) == 0
        assert len(test) == 0


class TestTemporalSplit:
    """Unit tests for TemporalSplit."""

    def test_default_ratios_approximate_proportions(self, synthetic_interactions_df):
        """Default 70/15/15 ratios produce approximately correct proportions."""
        splitter = TemporalSplit()
        train, val, test = splitter.split(synthetic_interactions_df, NUM_USERS, NUM_ITEMS)

        total = len(synthetic_interactions_df)
        assert abs(len(train) / total - 0.70) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05
        assert abs(len(test) / total - 0.15) < 0.05

    def test_rejects_ratios_not_summing_to_one(self):
        """Ratios that don't sum to 1.0 raise AssertionError."""
        with pytest.raises(AssertionError):
            TemporalSplit(train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)


class TestUserBasedSplit:
    """Unit tests for UserBasedSplit."""

    def test_each_user_in_exactly_one_split(self, synthetic_interactions_df):
        """Each user's interactions appear in exactly one split."""
        splitter = UserBasedSplit()
        train, val, test = splitter.split(synthetic_interactions_df, NUM_USERS, NUM_ITEMS)

        train_users = set(train["user_id"].unique())
        val_users = set(val["user_id"].unique())
        test_users = set(test["user_id"].unique())

        # Pairwise disjoint
        assert train_users & val_users == set()
        assert train_users & test_users == set()
        assert val_users & test_users == set()

        # Union equals all users
        all_users = set(synthetic_interactions_df["user_id"].unique())
        assert train_users | val_users | test_users == all_users

    def test_reproducibility_with_same_seed(self, synthetic_interactions_df):
        """Same seed produces identical splits."""
        s1 = UserBasedSplit(random_seed=123)
        s2 = UserBasedSplit(random_seed=123)

        t1, v1, te1 = s1.split(synthetic_interactions_df, NUM_USERS, NUM_ITEMS)
        t2, v2, te2 = s2.split(synthetic_interactions_df, NUM_USERS, NUM_ITEMS)

        pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
        pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))
        pd.testing.assert_frame_equal(te1.reset_index(drop=True), te2.reset_index(drop=True))


class TestGetSplitStrategy:
    """Unit tests for get_split_strategy factory."""

    @pytest.mark.parametrize("name,cls", [
        ("leave_one_out", LeaveOneOutSplit),
        ("temporal", TemporalSplit),
        ("user_based", UserBasedSplit),
    ])
    def test_valid_names(self, name, cls):
        strategy = get_split_strategy(name)
        assert isinstance(strategy, cls)

    @pytest.mark.parametrize("alias,cls", [
        ("loo", LeaveOneOutSplit),
        ("time", TemporalSplit),
        ("time_split", TemporalSplit),
        ("random_user_split", UserBasedSplit),
    ])
    def test_aliases(self, alias, cls):
        strategy = get_split_strategy(alias)
        assert isinstance(strategy, cls)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown split strategy"):
            get_split_strategy("nonexistent_strategy")



# ===================================================================
# 5.2–5.7  Split strategy property tests
# ===================================================================


# Feature: comprehensive-test-suite, Property 9: LeaveOneOutSplit correctness
@given(
    num_users=st.integers(min_value=1, max_value=8),
    items_per_user=st.integers(min_value=3, max_value=12),
)
@settings(max_examples=100)
def test_property_leave_one_out_correctness(num_users, items_per_user):
    """Property 9: For users with 3+ interactions, last→test, second-to-last→val."""
    df = _make_interactions_df(num_users, items_per_user)
    splitter = LeaveOneOutSplit(min_sequence_length=3)
    train, val, test = splitter.split(df, num_users, items_per_user)

    for uid in range(num_users):
        user_sorted = df[df["user_id"] == uid].sort_values("timestamp")
        items = user_sorted["item_id"].tolist()

        test_items = test[test["user_id"] == uid]["item_id"].tolist()
        val_items = val[val["user_id"] == uid]["item_id"].tolist()
        train_items = train[train["user_id"] == uid]["item_id"].tolist()

        assert test_items == [items[-1]], f"user {uid}: test mismatch"
        assert val_items == [items[-2]], f"user {uid}: val mismatch"
        assert sorted(train_items) == sorted(items[:-2]), f"user {uid}: train mismatch"


# Feature: comprehensive-test-suite, Property 10: TemporalSplit preserves approximate ratios
@given(
    num_users=st.integers(min_value=2, max_value=6),
    items_per_user=st.integers(min_value=5, max_value=15),
)
@settings(max_examples=100)
def test_property_temporal_split_ratios(num_users, items_per_user):
    """Property 10: TemporalSplit preserves approximate 70/15/15 ratios."""
    df = _make_interactions_df(num_users, items_per_user)
    total = len(df)
    assume(total >= 10)  # need enough rows for meaningful ratios

    splitter = TemporalSplit()
    train, val, test = splitter.split(df, num_users, items_per_user)

    # Each split size should be within ±1 row of the expected proportion
    assert abs(len(train) - int(total * 0.7)) <= 1
    assert abs(len(val) - int(total * 0.85) + int(total * 0.7)) <= 1
    assert len(train) + len(val) + len(test) == total


# Feature: comprehensive-test-suite, Property 11: TemporalSplit rejects invalid ratios
@given(
    train_r=st.floats(min_value=0.01, max_value=0.98, allow_nan=False),
    val_r=st.floats(min_value=0.01, max_value=0.98, allow_nan=False),
    test_r=st.floats(min_value=0.01, max_value=0.98, allow_nan=False),
)
@settings(max_examples=100)
def test_property_temporal_split_rejects_invalid_ratios(train_r, val_r, test_r):
    """Property 11: Ratios not summing to 1.0 are rejected."""
    assume(abs(train_r + val_r + test_r - 1.0) > 1e-6)
    with pytest.raises(AssertionError):
        TemporalSplit(train_ratio=train_r, val_ratio=val_r, test_ratio=test_r)


# Feature: comprehensive-test-suite, Property 12: UserBasedSplit disjoint assignment
@given(
    num_users=st.integers(min_value=3, max_value=10),
    items_per_user=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=100)
def test_property_user_based_split_disjoint(num_users, items_per_user):
    """Property 12: Each user is assigned to exactly one split."""
    df = _make_interactions_df(num_users, items_per_user)
    splitter = UserBasedSplit()
    train, val, test = splitter.split(df, num_users, items_per_user)

    train_users = set(train["user_id"].unique()) if len(train) > 0 else set()
    val_users = set(val["user_id"].unique()) if len(val) > 0 else set()
    test_users = set(test["user_id"].unique()) if len(test) > 0 else set()

    assert train_users & val_users == set()
    assert train_users & test_users == set()
    assert val_users & test_users == set()
    assert train_users | val_users | test_users == set(range(num_users))


# Feature: comprehensive-test-suite, Property 13: UserBasedSplit is reproducible
@given(
    seed=st.integers(min_value=0, max_value=10000),
    num_users=st.integers(min_value=3, max_value=8),
)
@settings(max_examples=100)
def test_property_user_based_split_reproducible(seed, num_users):
    """Property 13: Same seed → identical splits."""
    df = _make_interactions_df(num_users, 5)
    s1 = UserBasedSplit(random_seed=seed)
    s2 = UserBasedSplit(random_seed=seed)

    t1, v1, te1 = s1.split(df, num_users, 5)
    t2, v2, te2 = s2.split(df, num_users, 5)

    pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
    pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))
    pd.testing.assert_frame_equal(te1.reset_index(drop=True), te2.reset_index(drop=True))


# Feature: comprehensive-test-suite, Property 14: get_split_strategy rejects unknown names
@given(name=st.text(min_size=1, max_size=30).filter(lambda s: s not in SPLIT_STRATEGIES))
@settings(max_examples=100)
def test_property_get_split_strategy_rejects_unknown(name):
    """Property 14: Unknown strategy names raise ValueError."""
    with pytest.raises(ValueError):
        get_split_strategy(name)


# ===================================================================
# 5.8  Sequential dataset and sequence preparation unit tests
# ===================================================================


class TestPrepareSequences:
    """Unit tests for prepare_sequences."""

    def test_padding_to_max_seq_length(self):
        """Sequences are padded to max_seq_length with zeros."""
        df = pd.DataFrame([
            {"user_id": 0, "item_id": 1, "timestamp": 1},
            {"user_id": 0, "item_id": 2, "timestamp": 2},
            {"user_id": 0, "item_id": 3, "timestamp": 3},
        ])
        seqs = prepare_sequences(df, max_seq_length=5)
        assert len(seqs) == 1
        assert len(seqs[0]["sequence"]) == 5
        # Last two positions should be zero-padded
        assert seqs[0]["sequence"][-1] == 0
        assert seqs[0]["sequence"][-2] == 0

    def test_truncation_to_most_recent(self):
        """When user has more items than max_seq_length, keep most recent."""
        df = pd.DataFrame([
            {"user_id": 0, "item_id": i, "timestamp": i} for i in range(1, 12)
        ])
        seqs = prepare_sequences(df, max_seq_length=5)
        assert len(seqs[0]["sequence"]) == 5
        # Should keep items 7-11 (most recent 5)
        assert seqs[0]["sequence"] == [7, 8, 9, 10, 11]

    def test_gru4rec_target_extraction(self):
        """model_type='gru4rec': target is last item, excluded from sequence."""
        df = pd.DataFrame([
            {"user_id": 0, "item_id": 1, "timestamp": 1},
            {"user_id": 0, "item_id": 2, "timestamp": 2},
            {"user_id": 0, "item_id": 3, "timestamp": 3},
        ])
        seqs = prepare_sequences(df, max_seq_length=5, model_type="gru4rec")
        assert seqs[0]["target"] == 3
        # Sequence should not contain the target item at a non-padded position
        non_padded = [x for x in seqs[0]["sequence"] if x != 0]
        assert 3 not in non_padded

    def test_for_val_loo_builds_from_train(self):
        """for_val_loo=True builds sequences from train_df with target from val/test."""
        train_df = pd.DataFrame([
            {"user_id": 0, "item_id": 1, "timestamp": 1},
            {"user_id": 0, "item_id": 2, "timestamp": 2},
            {"user_id": 0, "item_id": 3, "timestamp": 3},
        ])
        val_df = pd.DataFrame([
            {"user_id": 0, "item_id": 4, "timestamp": 4},
        ])
        seqs = prepare_sequences(
            val_df, max_seq_length=5, for_val_loo=True, train_df=train_df
        )
        assert len(seqs) == 1
        assert seqs[0]["target"] == 4
        non_padded = [x for x in seqs[0]["sequence"] if x != 0]
        assert sorted(non_padded) == [1, 2, 3]


class TestBuildUserHistories:
    """Unit tests for build_user_histories."""

    def test_returns_correct_mapping(self, synthetic_interactions_df):
        histories = build_user_histories(synthetic_interactions_df)
        assert len(histories) == NUM_USERS
        for uid in range(NUM_USERS):
            expected = set(
                synthetic_interactions_df[
                    synthetic_interactions_df["user_id"] == uid
                ]["item_id"]
            )
            assert histories[uid] == expected


class TestSequentialDataset:
    """Unit tests for SequentialDataset."""

    def test_getitem_returns_correct_keys(self):
        seqs = [{"user_id": 0, "sequence": [1, 2, 0], "sequence_length": 2}]
        ds = SequentialDataset(seqs, max_seq_length=3)
        sample = ds[0]
        assert "user_id" in sample
        assert "sequence" in sample
        assert "sequence_length" in sample
        assert isinstance(sample["user_id"], torch.Tensor)
        assert isinstance(sample["sequence"], torch.Tensor)
        assert isinstance(sample["sequence_length"], torch.Tensor)

    def test_len_returns_correct_count(self):
        seqs = [
            {"user_id": i, "sequence": [1, 0], "sequence_length": 1}
            for i in range(7)
        ]
        ds = SequentialDataset(seqs, max_seq_length=2)
        assert len(ds) == 7


# ===================================================================
# 5.9–5.12  Sequential dataset property tests
# ===================================================================


# Feature: comprehensive-test-suite, Property 15: Sequence padding and truncation
@given(
    num_items=st.integers(min_value=1, max_value=20),
    max_seq=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100)
def test_property_sequence_padding_truncation(num_items, max_seq):
    """Property 15: Every sequence has exactly max_seq_length elements, zero-padded."""
    df = pd.DataFrame([
        {"user_id": 0, "item_id": i, "timestamp": i} for i in range(1, num_items + 1)
    ])
    seqs = prepare_sequences(df, max_seq_length=max_seq)
    if len(seqs) == 0:
        return
    seq = seqs[0]["sequence"]
    assert len(seq) == max_seq
    # If user had fewer items than max_seq, trailing positions are 0
    actual_len = min(num_items, max_seq)
    for pos in range(actual_len, max_seq):
        assert seq[pos] == 0


# Feature: comprehensive-test-suite, Property 16: GRU4Rec target extraction
@given(num_items=st.integers(min_value=2, max_value=15))
@settings(max_examples=100)
def test_property_gru4rec_target(num_items):
    """Property 16: GRU4Rec target is last item, not in sequence."""
    df = pd.DataFrame([
        {"user_id": 0, "item_id": i, "timestamp": i} for i in range(1, num_items + 1)
    ])
    seqs = prepare_sequences(df, max_seq_length=max(num_items, 5), model_type="gru4rec")
    if len(seqs) == 0:
        return
    s = seqs[0]
    assert s["target"] == num_items  # last item
    non_padded = [x for x in s["sequence"] if x != 0]
    assert s["target"] not in non_padded


# Feature: comprehensive-test-suite, Property 17: build_user_histories correctness
@given(
    num_users=st.integers(min_value=1, max_value=5),
    items_per_user=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_property_build_user_histories(num_users, items_per_user):
    """Property 17: Each user maps to the set of their interacted item IDs."""
    df = _make_interactions_df(num_users, items_per_user)
    histories = build_user_histories(df)
    for uid in range(num_users):
        expected = set(df[df["user_id"] == uid]["item_id"])
        assert histories[uid] == expected


# Feature: comprehensive-test-suite, Property 18: SequentialDataset contract
@given(num_seqs=st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_property_sequential_dataset_contract(num_seqs):
    """Property 18: __getitem__ returns correct keys as tensors, __len__ matches."""
    max_seq = 5
    seqs = [
        {"user_id": i, "sequence": list(range(1, max_seq + 1)), "sequence_length": max_seq}
        for i in range(num_seqs)
    ]
    ds = SequentialDataset(seqs, max_seq_length=max_seq)
    assert len(ds) == num_seqs
    for idx in range(num_seqs):
        sample = ds[idx]
        assert isinstance(sample["user_id"], torch.Tensor)
        assert isinstance(sample["sequence"], torch.Tensor)
        assert isinstance(sample["sequence_length"], torch.Tensor)


# ===================================================================
# 5.13  Implicit dataset unit tests
# ===================================================================


class TestPrepareImplicitInteractions:
    """Unit tests for prepare_implicit_interactions."""

    def test_all_rows_when_no_implicit_column(self):
        """All rows treated as positive when no 'implicit' column."""
        df = pd.DataFrame([
            {"user_id": 0, "item_id": 1},
            {"user_id": 0, "item_id": 2},
            {"user_id": 1, "item_id": 3},
        ])
        interactions = prepare_implicit_interactions(df)
        assert len(interactions) == 3

    def test_filters_to_implicit_eq_1(self):
        """Only rows with implicit==1 are included."""
        df = pd.DataFrame([
            {"user_id": 0, "item_id": 1, "implicit": 1},
            {"user_id": 0, "item_id": 2, "implicit": 0},
            {"user_id": 1, "item_id": 3, "implicit": 1},
        ])
        interactions = prepare_implicit_interactions(df)
        assert len(interactions) == 2

    def test_item_id_decremented_by_1(self):
        """item_id is converted from 1-indexed to 0-indexed."""
        df = pd.DataFrame([
            {"user_id": 0, "item_id": 5},
            {"user_id": 1, "item_id": 10},
        ])
        interactions = prepare_implicit_interactions(df)
        assert interactions[0]["item_id"] == 4
        assert interactions[1]["item_id"] == 9


class TestImplicitDataset:
    """Unit tests for ImplicitDataset."""

    def test_getitem_returns_correct_keys(self):
        interactions = [{"user_id": 0, "item_id": 3}, {"user_id": 1, "item_id": 7}]
        ds = ImplicitDataset(interactions)
        sample = ds[0]
        assert "user_id" in sample
        assert "item_id" in sample
        assert isinstance(sample["user_id"], torch.Tensor)
        assert isinstance(sample["item_id"], torch.Tensor)

    def test_len(self):
        interactions = [{"user_id": i, "item_id": i} for i in range(5)]
        ds = ImplicitDataset(interactions)
        assert len(ds) == 5


# ===================================================================
# 5.14–5.15  Implicit dataset property tests
# ===================================================================


# Feature: comprehensive-test-suite, Property 19: Implicit interaction filtering and index conversion
@given(
    num_rows=st.integers(min_value=1, max_value=20),
    has_implicit_col=st.booleans(),
)
@settings(max_examples=100)
def test_property_implicit_filtering_and_index(num_rows, has_implicit_col):
    """Property 19: Filtering and item_id decrement by 1."""
    data = {
        "user_id": [i % 3 for i in range(num_rows)],
        "item_id": [i + 1 for i in range(num_rows)],  # 1-indexed
    }
    if has_implicit_col:
        data["implicit"] = [1 if i % 2 == 0 else 0 for i in range(num_rows)]

    df = pd.DataFrame(data)
    interactions = prepare_implicit_interactions(df)

    if has_implicit_col:
        expected_count = sum(1 for i in range(num_rows) if i % 2 == 0)
    else:
        expected_count = num_rows
    assert len(interactions) == expected_count

    # Every output item_id should be input item_id - 1
    for inter in interactions:
        assert inter["item_id"] >= 0  # 0-indexed


# Feature: comprehensive-test-suite, Property 20: ImplicitDataset output format
@given(num_interactions=st.integers(min_value=1, max_value=15))
@settings(max_examples=100)
def test_property_implicit_dataset_format(num_interactions):
    """Property 20: __getitem__ returns user_id and item_id as tensors."""
    interactions = [{"user_id": i, "item_id": i + 1} for i in range(num_interactions)]
    ds = ImplicitDataset(interactions)
    assert len(ds) == num_interactions
    for idx in range(num_interactions):
        sample = ds[idx]
        assert isinstance(sample["user_id"], torch.Tensor)
        assert isinstance(sample["item_id"], torch.Tensor)


# ===================================================================
# 5.16  Sampler unit tests
# ===================================================================


class TestUniformSampler:
    """Unit tests for UniformSampler."""

    def test_excludes_positive_items(self):
        sampler = UniformSampler(num_items=20, num_negatives=5, item_offset=3)
        positives = {3, 5, 7}
        samples = sampler.sample(positives, 10)
        for s in samples:
            assert s not in positives

    def test_returns_requested_count(self):
        sampler = UniformSampler(num_items=20, num_negatives=5, item_offset=3)
        samples = sampler.sample(set(), 8)
        assert len(samples) == 8


class TestPopularitySampler:
    """Unit tests for PopularitySampler."""

    def test_excludes_positive_items(self):
        train_df = pd.DataFrame({
            "user_id": [0] * 10,
            "item_id": list(range(3, 13)),
        })
        sampler = PopularitySampler(
            num_items=20, num_negatives=5, train_df=train_df, item_offset=3
        )
        positives = {3, 5, 7}
        samples = sampler.sample(positives, 10)
        for s in samples:
            assert s not in positives

    def test_probability_distribution_sums_to_one(self):
        train_df = pd.DataFrame({
            "user_id": [0] * 10,
            "item_id": list(range(3, 13)),
        })
        sampler = PopularitySampler(
            num_items=20, num_negatives=5, train_df=train_df, item_offset=3
        )
        assert abs(sampler.probs.sum() - 1.0) < 1e-6


# ===================================================================
# 5.17–5.19  Sampler property tests
# ===================================================================


# Feature: comprehensive-test-suite, Property 23: Negative samplers exclude positive items
@given(
    num_positives=st.integers(min_value=1, max_value=5),
    n_samples=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_property_samplers_exclude_positives(num_positives, n_samples):
    """Property 23: Negative samplers never return positive items."""
    num_items = 20
    item_offset = 3
    positives = set(range(item_offset, item_offset + num_positives))

    # UniformSampler
    us = UniformSampler(num_items=num_items, num_negatives=5, item_offset=item_offset)
    u_samples = us.sample(positives, n_samples)
    for s in u_samples:
        assert s not in positives

    # PopularitySampler
    train_df = pd.DataFrame({
        "user_id": [0] * num_items,
        "item_id": list(range(item_offset, item_offset + num_items)),
    })
    ps = PopularitySampler(
        num_items=num_items, num_negatives=5, train_df=train_df, item_offset=item_offset
    )
    p_samples = ps.sample(positives, n_samples)
    for s in p_samples:
        assert s not in positives


# Feature: comprehensive-test-suite, Property 24: UniformSampler returns requested count
@given(n=st.integers(min_value=1, max_value=15))
@settings(max_examples=100)
def test_property_uniform_sampler_count(n):
    """Property 24: UniformSampler returns exactly n samples."""
    sampler = UniformSampler(num_items=20, num_negatives=5, item_offset=3)
    samples = sampler.sample(set(), n)
    assert len(samples) == n


# Feature: comprehensive-test-suite, Property 25: PopularitySampler distribution sums to 1
@given(num_items=st.integers(min_value=5, max_value=30))
@settings(max_examples=100)
def test_property_popularity_sampler_probs_sum(num_items):
    """Property 25: PopularitySampler probability distribution sums to 1.0."""
    item_offset = 3
    train_df = pd.DataFrame({
        "user_id": [0] * num_items,
        "item_id": list(range(item_offset, item_offset + num_items)),
    })
    sampler = PopularitySampler(
        num_items=num_items, num_negatives=5, train_df=train_df, item_offset=item_offset
    )
    assert abs(sampler.probs.sum() - 1.0) < 1e-6


# ===================================================================
# 5.20  Collate function unit tests
# ===================================================================


class TestSequentialNegativeSamplingCollate:
    """Unit tests for SequentialNegativeSamplingCollate."""

    def test_output_shape(self):
        num_items = 20
        num_neg = 4
        seq_len = 10
        batch_size = 3
        collate = SequentialNegativeSamplingCollate(
            num_items=num_items, num_negatives=num_neg
        )
        batch = _make_sequential_batch(batch_size=batch_size, seq_len=seq_len)
        result = collate(batch)
        assert result["neg_items"].shape == (batch_size, seq_len, num_neg)

    def test_negatives_exclude_user_history(self):
        num_items = 20
        user_histories = {0: {3, 4, 5}, 1: {6, 7}}
        collate = SequentialNegativeSamplingCollate(
            num_items=num_items, num_negatives=4, user_histories=user_histories
        )
        batch = _make_sequential_batch(batch_size=2, seq_len=5)
        result = collate(batch)
        for i, uid in enumerate([0, 1]):
            negs = result["neg_items"][i].flatten().tolist()
            for n in negs:
                if n != 0:  # skip padding
                    assert n not in user_histories.get(uid, set())


class TestImplicitNegativeSamplingCollate:
    """Unit tests for ImplicitNegativeSamplingCollate."""

    def test_output_shape(self):
        num_items = 20
        num_neg = 5
        batch_size = 4
        collate = ImplicitNegativeSamplingCollate(
            num_items=num_items, num_negatives=num_neg
        )
        batch = _make_implicit_batch(batch_size=batch_size)
        result = collate(batch)
        assert result["neg_items"].shape == (batch_size, num_neg)


class TestBatchSharedNegativeSamplingCollate:
    """Unit tests for BatchSharedNegativeSamplingCollate."""

    def test_output_shape(self):
        num_items = 20
        num_neg = 6
        batch_size = 3
        collate = BatchSharedNegativeSamplingCollate(
            num_items=num_items, num_negatives=num_neg
        )
        batch = _make_sequential_batch(batch_size=batch_size, seq_len=5)
        result = collate(batch)
        assert result["neg_items"].shape == (batch_size, num_neg)


class TestGraphNegativeSamplingCollate:
    """Unit tests for GraphNegativeSamplingCollate."""

    def test_output_shape(self):
        num_items = 20
        num_neg = 4
        batch_size = 3
        collate = GraphNegativeSamplingCollate(
            num_items=num_items, num_negatives=num_neg
        )
        batch = _make_graph_batch(batch_size=batch_size)
        result = collate(batch)
        assert result["neg_items"].shape == (batch_size, num_neg)

    def test_negatives_exclude_user_history(self):
        user_histories = {0: {3, 4}, 1: {5, 6}, 2: {7}}
        collate = GraphNegativeSamplingCollate(
            num_items=20, num_negatives=4, user_histories=user_histories
        )
        batch = _make_graph_batch(batch_size=3)
        result = collate(batch)
        for i, uid in enumerate([0, 1, 2]):
            negs = result["neg_items"][i].tolist()
            for n in negs:
                if n != 0:
                    assert n not in user_histories.get(uid, set())


# ===================================================================
# 5.21–5.22  Collate property tests
# ===================================================================


# Feature: comprehensive-test-suite, Property 26: Collate functions produce correct output shapes
@given(
    batch_size=st.integers(min_value=1, max_value=6),
    num_neg=st.integers(min_value=1, max_value=8),
    seq_len=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100)
def test_property_collate_output_shapes(batch_size, num_neg, seq_len):
    """Property 26: Collate functions produce correct output shapes."""
    num_items = 30

    # SequentialNegativeSamplingCollate → [batch, seq_len, num_neg]
    seq_collate = SequentialNegativeSamplingCollate(num_items=num_items, num_negatives=num_neg)
    seq_batch = _make_sequential_batch(batch_size=batch_size, seq_len=seq_len, num_items=num_items)
    seq_result = seq_collate(seq_batch)
    assert seq_result["neg_items"].shape == (batch_size, seq_len, num_neg)

    # ImplicitNegativeSamplingCollate → [batch, num_neg]
    imp_collate = ImplicitNegativeSamplingCollate(num_items=num_items, num_negatives=num_neg)
    imp_batch = _make_implicit_batch(batch_size=batch_size)
    imp_result = imp_collate(imp_batch)
    assert imp_result["neg_items"].shape == (batch_size, num_neg)

    # BatchSharedNegativeSamplingCollate → [batch, num_neg]
    bs_collate = BatchSharedNegativeSamplingCollate(num_items=num_items, num_negatives=num_neg)
    bs_batch = _make_sequential_batch(batch_size=batch_size, seq_len=seq_len, num_items=num_items)
    bs_result = bs_collate(bs_batch)
    assert bs_result["neg_items"].shape == (batch_size, num_neg)

    # GraphNegativeSamplingCollate → [batch, num_neg]
    gr_collate = GraphNegativeSamplingCollate(num_items=num_items, num_negatives=num_neg)
    gr_batch = _make_graph_batch(batch_size=batch_size, num_items=num_items)
    gr_result = gr_collate(gr_batch)
    assert gr_result["neg_items"].shape == (batch_size, num_neg)


# Feature: comprehensive-test-suite, Property 27: Collate negative samples exclude user history
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    num_neg=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=100)
def test_property_collate_negatives_exclude_history(batch_size, num_neg):
    """Property 27: Negative samples don't overlap with user positive history."""
    num_items = 30
    item_offset = 3
    # Build user histories with a few positives each
    user_histories = {i: {item_offset + i, item_offset + i + 1} for i in range(batch_size)}

    # Test with SequentialNegativeSamplingCollate
    collate = SequentialNegativeSamplingCollate(
        num_items=num_items, num_negatives=num_neg, user_histories=user_histories
    )
    batch = _make_sequential_batch(batch_size=batch_size, seq_len=5, num_items=num_items)
    result = collate(batch)
    for i in range(batch_size):
        negs = result["neg_items"][i].flatten().tolist()
        for n in negs:
            if n != 0:
                assert n not in user_histories[i]


# ===================================================================
# 5.23  Augmentation unit tests
# ===================================================================


class TestSequenceAugmenter:
    """Unit tests for SequenceAugmenter."""

    def test_crop_returns_valid_length(self):
        """Crop returns length between min_len and original."""
        aug = SequenceAugmenter(crop_prob=1.0)
        seq = [1, 2, 3, 4, 5, 6, 7, 8]
        random.seed(42)
        result = aug.crop(seq, min_len=2)
        assert 2 <= len(result) <= len(seq)

    def test_mask_places_mask_token(self):
        """Mask places mask_token at masked positions."""
        aug = SequenceAugmenter(mask_prob=1.0)
        seq = [1, 2, 3, 4, 5]
        random.seed(42)
        result = aug.mask(seq, mask_token=2)
        # At least one position should be mask_token
        assert 2 in result
        # Length preserved
        assert len(result) == len(seq)

    def test_reorder_preserves_multiset(self):
        """Reorder preserves multiset equality."""
        aug = SequenceAugmenter(reorder_prob=1.0)
        seq = [1, 2, 3, 4, 5, 6]
        random.seed(42)
        result = aug.reorder(seq, window=3)
        assert sorted(result) == sorted(seq)

    def test_augment_returns_valid_items(self):
        """Augment returns only valid items or mask token."""
        aug = SequenceAugmenter(crop_prob=1.0, mask_prob=1.0, reorder_prob=1.0)
        seq = [10, 20, 30, 40, 50]
        valid_set = set(seq) | {2}  # mask_token=2
        random.seed(42)
        for _ in range(20):
            result = aug.augment(seq)
            for item in result:
                assert item in valid_set


# ===================================================================
# 5.24–5.25  Augmentation property tests
# ===================================================================


# Feature: comprehensive-test-suite, Property 28: Reorder augmentation preserves multiset
@given(
    seq=st.lists(st.integers(min_value=1, max_value=100), min_size=4, max_size=20),
)
@settings(max_examples=100)
def test_property_reorder_preserves_multiset(seq):
    """Property 28: Reorder preserves multiset equality."""
    aug = SequenceAugmenter(reorder_prob=1.0)
    result = aug.reorder(seq, window=3)
    assert sorted(result) == sorted(seq)


# Feature: comprehensive-test-suite, Property 29: Augmentation produces valid items
@given(
    seq=st.lists(st.integers(min_value=3, max_value=100), min_size=3, max_size=15),
)
@settings(max_examples=100)
def test_property_augmentation_valid_items(seq):
    """Property 29: Augmentation produces only valid items or mask token."""
    aug = SequenceAugmenter(crop_prob=1.0, mask_prob=1.0, reorder_prob=1.0)
    mask_token = 2
    valid_set = set(seq) | {mask_token}
    result = aug.augment(seq)
    for item in result:
        assert item in valid_set


# ===================================================================
# 5.26  Cache unit tests
# ===================================================================


class TestDatasetCache:
    """Unit tests for DatasetCache."""

    def test_set_get_round_trip(self, tmp_path):
        """set/get round-trip returns equal data."""
        cache = DatasetCache(cache_dir=str(tmp_path / "cache"))
        data = {"key": [1, 2, 3], "value": "hello"}
        cache.set(data, dataset="test", split="train")
        retrieved = cache.get(dataset="test", split="train")
        assert retrieved == data

    def test_cache_miss_returns_none(self, tmp_path):
        """get with uncached kwargs returns None."""
        cache = DatasetCache(cache_dir=str(tmp_path / "cache"))
        assert cache.get(dataset="nonexistent") is None

    def test_cache_key_order_independent(self, tmp_path):
        """_get_cache_key produces same key regardless of kwarg order."""
        cache = DatasetCache(cache_dir=str(tmp_path / "cache"))
        key1 = cache._get_cache_key(a=1, b=2, c=3)
        key2 = cache._get_cache_key(c=3, a=1, b=2)
        assert key1 == key2


# ===================================================================
# 5.27–5.29  Cache property tests
# ===================================================================


# Feature: comprehensive-test-suite, Property 30: Cache round-trip
@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=20),
    key_val=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=100)
def test_property_cache_round_trip(data, key_val):
    """Property 30: set followed by get returns equal data."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cache = DatasetCache(cache_dir=td)
        cache.set(data, key=key_val)
        retrieved = cache.get(key=key_val)
        assert retrieved == data


# Feature: comprehensive-test-suite, Property 31: Cache miss returns None
@given(key_val=st.integers(min_value=0, max_value=10000))
@settings(max_examples=100)
def test_property_cache_miss_none(key_val):
    """Property 31: get with uncached kwargs returns None."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cache = DatasetCache(cache_dir=td)
        assert cache.get(key=key_val) is None


# Feature: comprehensive-test-suite, Property 32: Cache key is order-independent
@given(
    a=st.integers(min_value=0, max_value=100),
    b=st.text(min_size=1, max_size=10),
    c=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_cache_key_order_independent(a, b, c):
    """Property 32: Cache key is the same regardless of kwarg order."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cache = DatasetCache(cache_dir=td)
        key1 = cache._get_cache_key(a=a, b=b, c=c)
        key2 = cache._get_cache_key(c=c, a=a, b=b)
        key3 = cache._get_cache_key(b=b, c=c, a=a)
        assert key1 == key2 == key3


# ===================================================================
# ===================================================================
# Tests from test_dataset_extras.py
# Covers: GraphDataset, TraditionalDataModule, BaseDataset, datasets/constants.
# ===================================================================
# ===================================================================

import numpy as np
import scipy.sparse as sp
from unittest.mock import MagicMock

from rec_arena.datasets.graph_dataset import GraphDataset, to_graph
from rec_arena.datasets.traditional_datamodule import TraditionalDataModule
from rec_arena.datasets.constants import PAD_TOKEN, MASK_TOKEN, UNK_TOKEN, FIRST_ITEM_ID


# ===================================================================
# GraphDataset
# ===================================================================


class TestGraphDataset:
    @pytest.fixture
    def dataset(self):
        interactions = [
            {"user_id": 0, "item_id": 3, "label": 1},
            {"user_id": 1, "item_id": 4, "label": 1},
            {"user_id": 2, "item_id": 5, "label": 0},
        ]
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        return GraphDataset(interactions, edge_index)

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem_keys(self, dataset):
        item = dataset[0]
        assert "user_id" in item
        assert "item_id" in item
        assert "label" in item
        assert "edge_index" in item

    def test_getitem_types(self, dataset):
        item = dataset[0]
        assert item["user_id"].dtype == torch.long
        assert item["item_id"].dtype == torch.long
        assert item["label"].dtype == torch.float
        assert item["edge_index"].dtype == torch.long

    def test_getitem_values(self, dataset):
        item = dataset[1]
        assert item["user_id"].item() == 1
        assert item["item_id"].item() == 4
        assert item["label"].item() == 1.0


class TestToGraph:
    def test_converts_positive_interactions(self):
        df = pd.DataFrame({
            "user_id": [0, 1, 2],
            "item_id": [3, 4, 5],
            "implicit": [1, 1, 0],
        })
        interactions = to_graph(df)
        assert len(interactions) == 2  # Only positive
        assert interactions[0]["label"] == 1

    def test_empty_df(self):
        df = pd.DataFrame({"user_id": [], "item_id": [], "implicit": []})
        interactions = to_graph(df)
        assert len(interactions) == 0


# ===================================================================
# TraditionalDataModule
# ===================================================================


class TestTraditionalDataModule:
    @pytest.fixture
    def mock_dataset(self):
        ds = MagicMock()
        ds.num_users = 5
        ds.num_items = 10
        train_df = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2],
            "item_id": [1, 2, 3, 4, 5],
            "rating": [5.0, 4.0, 3.0, 5.0, 2.0],
        })
        val_df = pd.DataFrame({
            "user_id": [0, 1],
            "item_id": [3, 5],
            "rating": [4.0, 5.0],
        })
        test_df = pd.DataFrame({
            "user_id": [0, 2],
            "item_id": [4, 6],
            "rating": [5.0, 3.0],
        })
        ds.split.return_value = (train_df, val_df, test_df)
        return ds

    def test_setup_creates_matrices(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        dm.setup()
        assert dm.train_matrix is not None
        assert dm.val_matrix is not None
        assert dm.test_matrix is not None

    def test_get_train_matrix_shape(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        dm.setup()
        m = dm.get_train_matrix()
        assert m.shape == (5, 10)

    def test_get_train_matrix_before_setup_raises(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        with pytest.raises(RuntimeError, match="setup"):
            dm.get_train_matrix()

    def test_get_val_matrix_before_setup_raises(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        with pytest.raises(RuntimeError, match="setup"):
            dm.get_val_matrix()

    def test_get_test_matrix_before_setup_raises(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        with pytest.raises(RuntimeError, match="setup"):
            dm.get_test_matrix()

    def test_get_train_df(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        dm.setup()
        df = dm.get_train_df()
        assert isinstance(df, pd.DataFrame)

    def test_get_val_df(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        dm.setup()
        df = dm.get_val_df()
        assert isinstance(df, pd.DataFrame)

    def test_get_test_df(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        dm.setup()
        df = dm.get_test_df()
        assert isinstance(df, pd.DataFrame)

    def test_get_train_df_before_setup_raises(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        with pytest.raises(RuntimeError, match="setup"):
            dm.get_train_df()

    def test_get_val_df_before_setup_raises(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        with pytest.raises(RuntimeError, match="setup"):
            dm.get_val_df()

    def test_get_test_df_before_setup_raises(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset)
        with pytest.raises(RuntimeError, match="setup"):
            dm.get_test_df()

    def test_implicit_threshold(self, mock_dataset):
        dm = TraditionalDataModule(mock_dataset, implicit_threshold=4.0)
        dm.setup()
        m = dm.get_train_matrix()
        # Only ratings >= 4.0 should be in the matrix
        assert sp.issparse(m)


# ===================================================================
# Constants
# ===================================================================


class TestConstants:
    def test_pad_token(self):
        assert PAD_TOKEN == 0

    def test_mask_token(self):
        assert MASK_TOKEN == 1

    def test_unk_token(self):
        assert UNK_TOKEN == 2

    def test_first_item_id(self):
        assert FIRST_ITEM_ID == 3


# ===================================================================
# ===================================================================
# Tests from test_base_dataset.py
# Covers: BaseDataset, LocalDataset — load_data, _filter_by_min_interactions,
# _remap_ids, split, get_id_mappings, _detect_format, _load_raw_data.
# ===================================================================
# ===================================================================

import os

from rec_arena.datasets.base_dataset import BaseDataset, LocalDataset


# ===================================================================
# Concrete stub for BaseDataset
# ===================================================================


class StubDataset(BaseDataset):
    """Concrete BaseDataset backed by an in-memory DataFrame."""

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self._raw_df = df

    def _get_default_split_type(self):
        return "leave_one_out"

    def _load_raw_data(self):
        return self._raw_df.copy()


def _make_df(num_users=5, items_per_user=10):
    rows = []
    for u in range(num_users):
        for rank, i in enumerate(range(items_per_user)):
            rows.append({"user_id": u + 100, "item_id": i + 200, "timestamp": u * 100 + rank})
    return pd.DataFrame(rows)


# ===================================================================
# BaseDataset
# ===================================================================


class TestBaseDatasetLoadData:
    def test_sets_num_users_and_items(self):
        df = _make_df(5, 10)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        assert ds.num_users == 5
        assert ds.num_items == 10

    def test_interactions_df_populated(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        assert ds.interactions_df is not None
        assert len(ds.interactions_df) == 15

    def test_implicit_threshold_filters(self):
        df = _make_df(3, 5)
        df["rating"] = [5.0, 4.0, 3.0, 2.0, 1.0] * 3
        ds = StubDataset(df, min_interactions=1, implicit_threshold=4.0)
        ds.load_data()
        # Only ratings >= 4.0 should remain
        assert len(ds.interactions_df) < 15

    def test_adds_implicit_column_when_missing(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        assert "implicit" in ds.interactions_df.columns


class TestBaseDatasetFilterByMinInteractions:
    def test_filters_users_below_threshold(self):
        # User 0 has 2 interactions, users 1-5 each have 10 interactions
        # Items need to also have >= 5 interactions to survive
        rows = [{"user_id": 0, "item_id": i, "timestamp": i} for i in range(2)]
        for u in range(1, 6):
            rows += [{"user_id": u, "item_id": i, "timestamp": u * 100 + i} for i in range(10)]
        df = pd.DataFrame(rows)
        ds = StubDataset(df, min_interactions=5)
        ds.load_data()
        # User 0 should be filtered out (only 2 interactions)
        assert 0 not in ds.interactions_df["user_id"].values or ds.num_users < 6

    def test_min_interactions_1_keeps_all(self):
        df = _make_df(5, 10)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        assert ds.num_users == 5


class TestBaseDatasetRemapIds:
    def test_users_are_zero_indexed(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        assert ds.interactions_df["user_id"].min() == 0

    def test_items_are_one_indexed(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        assert ds.interactions_df["item_id"].min() == 1

    def test_ids_are_contiguous(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        users = sorted(ds.interactions_df["user_id"].unique())
        assert users == list(range(3))
        items = sorted(ds.interactions_df["item_id"].unique())
        assert items == list(range(1, 6))


class TestBaseDatasetSplit:
    def test_split_returns_three_dataframes(self):
        df = _make_df(5, 10)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        train, val, test = ds.split()
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_auto_loads_data(self):
        df = _make_df(5, 10)
        ds = StubDataset(df, min_interactions=1)
        # Don't call load_data — split should do it
        train, val, test = ds.split()
        assert ds.interactions_df is not None

    def test_split_override_type(self):
        df = _make_df(5, 10)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        train, val, test = ds.split(split_type="temporal")
        assert len(train) + len(val) + len(test) == len(ds.interactions_df)


class TestBaseDatasetGetIdMappings:
    def test_returns_copies(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1)
        ds.load_data()
        user_map, item_map = ds.get_id_mappings()
        assert isinstance(user_map, dict)
        assert isinstance(item_map, dict)
        # Modifying returned maps shouldn't affect internal state
        user_map[999] = 999
        u2, _ = ds.get_id_mappings()
        assert 999 not in u2


class TestBaseDatasetSplitType:
    def test_default_split_type(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1)
        assert ds.split_type == "leave_one_out"

    def test_override_split_type(self):
        df = _make_df(3, 5)
        ds = StubDataset(df, min_interactions=1, split_type="temporal")
        assert ds.split_type == "temporal"


# ===================================================================
# LocalDataset
# ===================================================================


class TestLocalDatasetDetectFormat:
    def test_csv(self):
        ds = LocalDataset("data.csv", min_interactions=1)
        assert ds._detect_format() == "csv"

    def test_tsv(self):
        ds = LocalDataset("data.tsv", min_interactions=1)
        assert ds._detect_format() == "tsv"

    def test_parquet(self):
        ds = LocalDataset("data.parquet", min_interactions=1)
        assert ds._detect_format() == "parquet"

    def test_dat(self):
        ds = LocalDataset("data.dat", min_interactions=1)
        assert ds._detect_format() == "dat"

    def test_unknown_defaults_to_csv(self):
        ds = LocalDataset("data.xyz", min_interactions=1)
        assert ds._detect_format() == "csv"

    def test_explicit_format_overrides(self):
        ds = LocalDataset("data.csv", file_format="tsv", min_interactions=1)
        assert ds._detect_format() == "tsv"


class TestLocalDatasetLoadRawData:
    def test_load_csv(self, tmp_path):
        path = str(tmp_path / "data.csv")
        with open(path, "w") as f:
            f.write("user_id,item_id,timestamp\n0,10,1\n1,20,2\n")

        ds = LocalDataset(path, file_format="csv", separator=",", min_interactions=1)
        ds.load_data()
        assert ds.num_users == 2

    def test_load_tsv(self, tmp_path):
        df = pd.DataFrame({"user_id": [0, 1], "item_id": [10, 20], "timestamp": [1, 2]})
        path = str(tmp_path / "data.tsv")
        df.to_csv(path, index=False, sep="\t")

        ds = LocalDataset(path, file_format="tsv", min_interactions=1)
        ds.load_data()
        assert ds.num_users == 2

    def test_load_with_column_names(self, tmp_path):
        path = str(tmp_path / "data.csv")
        with open(path, "w") as f:
            f.write("0,10,5.0,1000\n1,20,4.0,2000\n")

        ds = LocalDataset(
            path, file_format="csv", separator=",",
            column_names=["user_id", "item_id", "rating", "timestamp"],
            min_interactions=1,
        )
        ds.load_data()
        assert ds.num_users == 2

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"a": [0], "b": [1]})
        path = str(tmp_path / "data.csv")
        df.to_csv(path, index=False)

        ds = LocalDataset(path, min_interactions=1)
        with pytest.raises(ValueError, match="user_id"):
            ds.load_data()

    def test_no_timestamp_creates_dummy(self, tmp_path):
        path = str(tmp_path / "data.csv")
        with open(path, "w") as f:
            f.write("user_id,item_id\n0,10\n0,20\n1,30\n")

        ds = LocalDataset(path, file_format="csv", separator=",", min_interactions=1)
        ds.load_data()
        assert "timestamp" in ds.interactions_df.columns

    def test_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "data.csv")
        pd.DataFrame({"user_id": [0], "item_id": [1]}).to_csv(path, index=False)
        ds = LocalDataset(path, file_format="xml", min_interactions=1)
        with pytest.raises(ValueError, match="Unsupported"):
            ds.load_data()

    def test_load_parquet(self, tmp_path):
        df = pd.DataFrame({"user_id": [0, 1], "item_id": [10, 20], "timestamp": [1, 2]})
        path = str(tmp_path / "data.parquet")
        df.to_parquet(path, index=False)

        ds = LocalDataset(path, min_interactions=1)
        ds.load_data()
        assert ds.num_users == 2

    def test_default_split_type_is_loo(self):
        ds = LocalDataset("data.csv", min_interactions=1)
        assert ds.split_type == "leave_one_out"
