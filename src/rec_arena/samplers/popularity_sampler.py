"""Popularity-based negative sampler."""

import numpy as np
from typing import Set
from .base import BaseSampler, DEFAULT_ITEM_OFFSET


class PopularitySampler(BaseSampler):
    """Samples popular items as negatives (harder negatives)."""
    
    def __init__(self, num_items: int, num_negatives: int = 50, seed: int = 42, item_offset: int = DEFAULT_ITEM_OFFSET):
        super().__init__(num_items, num_negatives, seed, item_offset)
        self.rng = np.random.default_rng(seed)
        self.popularity_scores = self._create_popularity_distribution()
        
    def _create_popularity_distribution(self):
        """Create a popularity distribution where some items are more popular."""
        np.random.seed(self.seed)
        popularity = np.random.zipf(1.5, self.num_items)
        popularity = popularity / popularity.sum()
        return popularity
    
    def sample(self, positive_items: Set[int], user_id: int = None) -> list:
        """Sample popular items as negatives."""
        all_items = set(range(self.item_offset, self.num_items + self.item_offset))
        negative_candidates = list(all_items - positive_items)
        
        if len(negative_candidates) == 0:
            return []
        
        # Map candidates to 0-indexed for popularity lookup
        candidate_indices = np.array(negative_candidates) - self.item_offset
        candidate_probs = self.popularity_scores[candidate_indices]
        candidate_probs = candidate_probs / candidate_probs.sum()
        
        num_to_sample = min(self.num_negatives, len(negative_candidates))
        sampled_negatives = self.rng.choice(
            negative_candidates, size=num_to_sample, replace=False, p=candidate_probs
        )
        return sampled_negatives.tolist()
    
    def sample_many(self, positive_items: Set[int], count: int, user_id: int = None) -> np.ndarray:
        """Sample count negatives with popularity weighting (with replacement)."""
        all_items = set(range(self.item_offset, self.num_items + self.item_offset))
        negative_candidates = np.array(list(all_items - positive_items))
        
        if len(negative_candidates) == 0:
            return np.zeros(count, dtype=np.int64)
        
        # Map candidates to 0-indexed for popularity lookup
        candidate_indices = negative_candidates - self.item_offset
        candidate_probs = self.popularity_scores[candidate_indices]
        candidate_probs = candidate_probs / candidate_probs.sum()
        
        return self.rng.choice(negative_candidates, size=count, replace=True, p=candidate_probs)