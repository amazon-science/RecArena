"""Base negative sampler interface."""

from abc import ABC, abstractmethod
from typing import List, Set
import numpy as np

# Default offset for item indices (PAD=0, UNK=1, MASK=2, items start at 3)
DEFAULT_ITEM_OFFSET = 3


class BaseSampler(ABC):
    """Abstract base class for negative samplers."""
    
    def __init__(self, num_items: int, num_negatives: int = 5, seed: int = 42, item_offset: int = DEFAULT_ITEM_OFFSET):
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.seed = seed
        self.item_offset = item_offset  # Items are in range [item_offset, num_items + item_offset - 1]
        self.is_fitted = False
        np.random.seed(seed)
    
    def fit(self, dataset, interactions_df=None):
        """Fit the sampler to the dataset (optional for data-dependent samplers).
        
        Args:
            dataset: The dataset object
            interactions_df: DataFrame with user-item interactions (optional)
        """
        self.is_fitted = True
        return self
    
    @abstractmethod
    def sample(self, positive_items: Set[int], user_id: int = None) -> List[int]:
        """Sample negative items for a user.
        
        Args:
            positive_items: Set of items the user has interacted with
            user_id: Optional user ID for user-specific sampling
            
        Returns:
            List of negative item IDs
        """
        pass
    
    def _check_fitted(self):
        """Check if sampler has been fitted (for data-dependent samplers)."""
        if hasattr(self, '_requires_fitting') and self._requires_fitting and not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} requires fitting before sampling. Call fit() first.")
    
    def sample_batch(self, positive_items_batch: List[Set[int]], user_ids: List[int] = None) -> List[List[int]]:
        """Sample negatives for a batch of users."""
        if user_ids is None:
            user_ids = [None] * len(positive_items_batch)
            
        return [
            self.sample(pos_items, user_id) 
            for pos_items, user_id in zip(positive_items_batch, user_ids)
        ]
    
    def sample_many(self, positive_items: Set[int], count: int, user_id: int = None) -> np.ndarray:
        """Sample count negatives at once (with replacement). Override for efficiency."""
        # Items are in range [item_offset, num_items + item_offset - 1]
        all_items = np.arange(self.item_offset, self.num_items + self.item_offset)
        candidates = np.setdiff1d(all_items, list(positive_items), assume_unique=True)
        if len(candidates) == 0:
            return np.zeros(count, dtype=np.int64)
        return np.random.choice(candidates, size=count, replace=True)