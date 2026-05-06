"""Negative sampling strategies for recommendation models."""

import numpy as np
import pandas as pd
from typing import Set


class UniformSampler:
    """Uniform random negative sampling (baseline)."""

    def __init__(self, num_items: int, num_negatives: int, item_offset: int = 3):
        self.num_negatives = num_negatives
        self.all_items = np.arange(item_offset, num_items + item_offset)
        self.rng = np.random.default_rng()

    def sample(self, user_positives: Set[int], n: int) -> np.ndarray:
        candidates = np.setdiff1d(self.all_items, list(user_positives), assume_unique=True)
        return self.rng.choice(candidates, size=n, replace=len(candidates) < n)


class PopularitySampler:
    """Popularity-based negative sampling: sample proportional to item frequency."""

    def __init__(
        self,
        num_items: int,
        num_negatives: int,
        train_df: pd.DataFrame,
        item_offset: int = 3,
        smoothing: float = 0.75,
    ):
        self.num_negatives = num_negatives
        self.item_offset = item_offset
        self.rng = np.random.default_rng()

        # Build popularity distribution over offset-adjusted item ids
        counts = train_df["item_id"].value_counts()
        all_items = np.arange(item_offset, num_items + item_offset)
        freqs = np.array([counts.get(i, 0) for i in all_items], dtype=np.float64)
        freqs = np.power(freqs + 1, smoothing)  # +1 to avoid zero for unseen items
        self.all_items = all_items
        self.probs = freqs / freqs.sum()

    def sample(self, user_positives: Set[int], n: int) -> np.ndarray:
        # Zero out positives and renormalize
        mask = np.ones(len(self.all_items), dtype=bool)
        for p in user_positives:
            idx = p - self.item_offset
            if 0 <= idx < len(mask):
                mask[idx] = False
        probs = self.probs * mask
        total = probs.sum()
        if total == 0:
            return self.rng.choice(self.all_items, size=n, replace=True)
        probs = probs / total
        return self.rng.choice(self.all_items, size=n, replace=True, p=probs)
