"""Tests for rec_arena.samplers module.

Covers: BaseSampler, RandomSampler, PopularitySampler, GraphRandomSampler,
HardNegativeSampler, CategorySampler, GenreDiverseSampler, GenreSimilarSampler.
"""

import numpy as np
import pandas as pd
import pytest

from rec_arena.samplers.base import BaseSampler
from rec_arena.samplers.random_sampler import RandomSampler
from rec_arena.samplers.popularity_sampler import PopularitySampler
from rec_arena.samplers.graph_random_sampler import GraphRandomSampler
from rec_arena.samplers.hard_negative_sampler import HardNegativeSampler
from rec_arena.samplers.category_sampler import CategorySampler
from rec_arena.samplers.genre_sampler import GenreDiverseSampler, GenreSimilarSampler

NUM_ITEMS = 50
NUM_NEG = 5
ITEM_OFFSET = 3


# ===================================================================
# RandomSampler
# ===================================================================


class TestRandomSampler:
    def test_sample_returns_correct_count(self):
        sampler = RandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items={3, 4, 5})
        assert len(negs) == NUM_NEG

    def test_sample_excludes_positives(self):
        positives = {3, 4, 5, 6, 7}
        sampler = RandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items=positives)
        assert all(n not in positives for n in negs)

    def test_sample_respects_item_offset(self):
        sampler = RandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42, item_offset=ITEM_OFFSET)
        negs = sampler.sample(positive_items={3, 4})
        assert all(n >= ITEM_OFFSET for n in negs)

    def test_sample_many(self):
        sampler = RandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample_many(positive_items={3, 4}, count=20)
        assert len(negs) == 20
        assert all(n not in {3, 4} for n in negs)

    def test_sample_many_empty_candidates(self):
        # All items are positive
        all_pos = set(range(ITEM_OFFSET, NUM_ITEMS + ITEM_OFFSET))
        sampler = RandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42, item_offset=ITEM_OFFSET)
        negs = sampler.sample_many(positive_items=all_pos, count=5)
        assert np.all(negs == 0)

    def test_large_positive_set_uses_rejection(self):
        # >100 positives triggers rejection sampling path
        positives = set(range(ITEM_OFFSET, ITEM_OFFSET + 150))
        sampler = RandomSampler(num_items=200, num_negatives=5, seed=42, item_offset=ITEM_OFFSET)
        negs = sampler.sample(positive_items=positives)
        assert len(negs) == 5
        assert all(n not in positives for n in negs)


# ===================================================================
# PopularitySampler
# ===================================================================


class TestPopularitySampler:
    def test_sample_returns_correct_count(self):
        sampler = PopularitySampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items={3, 4, 5})
        assert len(negs) == NUM_NEG

    def test_sample_excludes_positives(self):
        positives = {3, 4, 5}
        sampler = PopularitySampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items=positives)
        assert all(n not in positives for n in negs)

    def test_sample_empty_candidates(self):
        all_pos = set(range(ITEM_OFFSET, NUM_ITEMS + ITEM_OFFSET))
        sampler = PopularitySampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42, item_offset=ITEM_OFFSET)
        negs = sampler.sample(positive_items=all_pos)
        assert negs == []

    def test_sample_many(self):
        sampler = PopularitySampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample_many(positive_items={3}, count=20)
        assert len(negs) == 20

    def test_sample_many_empty_candidates(self):
        all_pos = set(range(ITEM_OFFSET, NUM_ITEMS + ITEM_OFFSET))
        sampler = PopularitySampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42, item_offset=ITEM_OFFSET)
        negs = sampler.sample_many(positive_items=all_pos, count=5)
        assert np.all(negs == 0)


# ===================================================================
# GraphRandomSampler
# ===================================================================


class TestGraphRandomSampler:
    def test_sample_returns_correct_count(self):
        sampler = GraphRandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items={0, 1, 2})
        assert len(negs) == NUM_NEG

    def test_sample_excludes_positives(self):
        positives = {0, 1, 2}
        sampler = GraphRandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items=positives)
        assert all(n not in positives for n in negs)

    def test_zero_indexed_by_default(self):
        sampler = GraphRandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items={0})
        assert all(n >= 0 for n in negs)

    def test_empty_positives(self):
        sampler = GraphRandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        negs = sampler.sample(positive_items=set())
        assert len(negs) == NUM_NEG

    def test_all_items_positive_returns_empty(self):
        sampler = GraphRandomSampler(num_items=5, num_negatives=3, seed=42)
        negs = sampler.sample(positive_items=set(range(5)))
        assert negs == []


# ===================================================================
# HardNegativeSampler
# ===================================================================


class TestHardNegativeSampler:
    @pytest.fixture
    def interactions_df(self):
        rows = []
        for uid in range(10):
            for iid in range(3, 13):
                rows.append({"user_id": uid, "item_id": iid, "rating": 1.0})
        return pd.DataFrame(rows)

    def test_requires_fitting(self):
        sampler = HardNegativeSampler(num_items=20, num_negatives=5)
        with pytest.raises(ValueError, match="requires fitting"):
            sampler.sample(positive_items={3, 4})

    def test_fit_and_sample(self, interactions_df):
        sampler = HardNegativeSampler(num_items=20, num_negatives=5, seed=42)
        sampler.fit(dataset=None, interactions_df=interactions_df)
        negs = sampler.sample(positive_items={3, 4, 5})
        assert len(negs) == 5
        assert all(n not in {3, 4, 5} for n in negs)

    def test_fit_without_df_raises(self):
        sampler = HardNegativeSampler(num_items=20, num_negatives=5)
        with pytest.raises(ValueError, match="interactions_df is required"):
            sampler.fit(dataset=object())


# ===================================================================
# CategorySampler
# ===================================================================


class TestCategorySampler:
    def test_requires_fitting(self):
        sampler = CategorySampler(num_items=20, num_negatives=5)
        with pytest.raises(ValueError, match="requires fitting"):
            sampler.sample(positive_items={3, 4})

    def test_fit_and_sample(self):
        item_categories = {
            3: ["action"], 4: ["action"], 5: ["comedy"],
            6: ["comedy"], 7: ["drama"], 8: ["drama"],
        }
        sampler = CategorySampler(num_items=20, num_negatives=3, seed=42)
        sampler.fit(dataset=None, item_categories=item_categories)
        negs = sampler.sample(positive_items={3, 4})
        assert len(negs) == 3
        assert all(n not in {3, 4} for n in negs)

    def test_empty_candidates_returns_empty(self):
        item_categories = {3: ["a"]}
        sampler = CategorySampler(num_items=1, num_negatives=5, seed=42, item_offset=3)
        sampler.fit(dataset=None, item_categories=item_categories)
        negs = sampler.sample(positive_items={3})
        assert negs == []


# ===================================================================
# GenreDiverseSampler
# ===================================================================


class TestGenreDiverseSampler:
    @pytest.fixture
    def item_genres(self):
        return {
            3: [0, 1], 4: [1, 2], 5: [2, 3],
            6: [4], 7: [5], 8: [0, 5],
        }

    def test_sample_returns_correct_count(self, item_genres):
        sampler = GenreDiverseSampler(
            num_items=10, item_genres=item_genres, num_negatives=3, seed=42
        )
        negs = sampler.sample(positive_items={3})
        assert len(negs) == 3

    def test_sample_excludes_positives(self, item_genres):
        sampler = GenreDiverseSampler(
            num_items=10, item_genres=item_genres, num_negatives=3, seed=42
        )
        negs = sampler.sample(positive_items={3, 4})
        assert all(n not in {3, 4} for n in negs)

    def test_sample_many(self, item_genres):
        sampler = GenreDiverseSampler(
            num_items=10, item_genres=item_genres, num_negatives=3, seed=42
        )
        negs = sampler.sample_many(positive_items={3}, count=10)
        assert len(negs) == 10

    def test_empty_candidates(self):
        sampler = GenreDiverseSampler(
            num_items=1, item_genres={3: [0]}, num_negatives=3, seed=42, item_offset=3
        )
        negs = sampler.sample(positive_items={3})
        assert negs == []


# ===================================================================
# GenreSimilarSampler
# ===================================================================


class TestGenreSimilarSampler:
    @pytest.fixture
    def item_genres(self):
        return {
            3: [0, 1], 4: [1, 2], 5: [2, 3],
            6: [4], 7: [5], 8: [0, 5],
        }

    def test_sample_returns_correct_count(self, item_genres):
        sampler = GenreSimilarSampler(
            num_items=10, item_genres=item_genres, num_negatives=3, seed=42
        )
        negs = sampler.sample(positive_items={3})
        assert len(negs) == 3

    def test_sample_excludes_positives(self, item_genres):
        sampler = GenreSimilarSampler(
            num_items=10, item_genres=item_genres, num_negatives=3, seed=42
        )
        negs = sampler.sample(positive_items={3, 4})
        assert all(n not in {3, 4} for n in negs)

    def test_sample_many(self, item_genres):
        sampler = GenreSimilarSampler(
            num_items=10, item_genres=item_genres, num_negatives=3, seed=42
        )
        negs = sampler.sample_many(positive_items={3}, count=10)
        assert len(negs) == 10

    def test_no_genre_info_falls_back(self, item_genres):
        sampler = GenreSimilarSampler(
            num_items=10, item_genres=item_genres, num_negatives=3, seed=42
        )
        # Positive items not in item_genres
        negs = sampler.sample(positive_items={20})
        assert len(negs) == 3


# ===================================================================
# BaseSampler
# ===================================================================


class TestBaseSampler:
    def test_sample_batch(self):
        sampler = RandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        batch_pos = [{3, 4}, {5, 6}, {7}]
        results = sampler.sample_batch(batch_pos)
        assert len(results) == 3
        assert all(len(r) == NUM_NEG for r in results)

    def test_sample_batch_with_user_ids(self):
        sampler = RandomSampler(num_items=NUM_ITEMS, num_negatives=NUM_NEG, seed=42)
        batch_pos = [{3, 4}, {5, 6}]
        results = sampler.sample_batch(batch_pos, user_ids=[0, 1])
        assert len(results) == 2
