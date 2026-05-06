"""Random negative sampler."""

from typing import List, Set
import numpy as np
from .base import BaseSampler, DEFAULT_ITEM_OFFSET


class RandomSampler(BaseSampler):
    """Random uniform negative sampling."""
    
    def __init__(self, num_items: int, num_negatives: int = 5, seed: int = None, item_offset: int = DEFAULT_ITEM_OFFSET):
        super().__init__(num_items, num_negatives, seed if seed is not None else 42, item_offset)
        self.rng = np.random.default_rng(seed)
    
    def sample(self, positive_items: Set[int], user_id: int = None) -> List[int]:
        """Sample negatives uniformly at random."""
        # Items are in range [item_offset, num_items + item_offset - 1]
        if len(positive_items) < 100:  # Small positive set
            all_items = set(range(self.item_offset, self.num_items + self.item_offset))
            negative_candidates = list(all_items - positive_items)
            num_to_sample = min(self.num_negatives, len(negative_candidates))
            return self.rng.choice(negative_candidates, size=num_to_sample, replace=False).tolist()
        else:  # Large positive set - use rejection sampling
            negatives = []
            while len(negatives) < self.num_negatives:
                candidate = self.rng.integers(self.item_offset, self.num_items + self.item_offset)
                if candidate not in positive_items:
                    negatives.append(candidate)
            return negatives
    
    def sample_many(self, positive_items: Set[int], count: int, user_id: int = None) -> np.ndarray:
        """Sample count negatives at once (with replacement)."""
        all_items = np.arange(self.item_offset, self.num_items + self.item_offset)
        candidates = np.setdiff1d(all_items, list(positive_items), assume_unique=True)
        if len(candidates) == 0:
            return np.zeros(count, dtype=np.int64)
        return self.rng.choice(candidates, size=count, replace=True)