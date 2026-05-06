"""Graph-specific random negative sampler."""

from typing import List, Set
import numpy as np
from .base import BaseSampler


class GraphRandomSampler(BaseSampler):
    """Random uniform negative sampling for graph models (0-indexed items).
    
    Note: Graph models typically use 0-indexed items, so item_offset=0 by default.
    """
    
    def __init__(self, num_items: int, num_negatives: int = 5, seed: int = None, item_offset: int = 0):
        super().__init__(num_items, num_negatives, seed if seed is not None else 42, item_offset)
        self.rng = np.random.default_rng(seed)
        # Pre-allocate arrays for efficiency
        self.all_items = np.arange(self.item_offset, self.num_items + self.item_offset, dtype=np.int32)
        self.mask = np.ones(self.num_items, dtype=bool)
    
    def sample(self, positive_items: Set[int], user_id: int = None) -> List[int]:
        """Sample negatives uniformly at random using fast NumPy operations."""
        if len(positive_items) == 0:
            num_to_sample = min(self.num_negatives, self.num_items)
            return self.rng.choice(self.all_items, size=num_to_sample, replace=False).tolist()
        
        if len(positive_items) >= self.num_items:
            return []
        
        # Fast boolean masking approach
        self.mask.fill(True)
        if positive_items:
            # Adjust for item_offset when indexing into mask
            pos_array = np.array(list(positive_items), dtype=np.int32) - self.item_offset
            valid_mask = (pos_array >= 0) & (pos_array < self.num_items)
            self.mask[pos_array[valid_mask]] = False
        
        negative_candidates = self.all_items[self.mask]
        
        if len(negative_candidates) == 0:
            return []
        
        num_to_sample = min(self.num_negatives, len(negative_candidates))
        return self.rng.choice(negative_candidates, size=num_to_sample, replace=False).tolist()