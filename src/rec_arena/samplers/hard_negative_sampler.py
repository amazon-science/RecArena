"""Hard negative sampler using item similarity."""

import numpy as np
import pandas as pd
from typing import Set
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseSampler, DEFAULT_ITEM_OFFSET


class HardNegativeSampler(BaseSampler):
    """Samples items similar to user's positive items (harder negatives)."""
    
    def __init__(self, num_items: int, num_negatives: int = 50, seed: int = 42, 
                 similarity_threshold: float = 0.1, item_offset: int = DEFAULT_ITEM_OFFSET):
        super().__init__(num_items, num_negatives, seed, item_offset)
        self.similarity_threshold = similarity_threshold
        self.item_similarity_matrix = None
        self._requires_fitting = True
        
    def fit(self, dataset, interactions_df=None):
        """Fit the sampler by computing item-item similarity matrix."""
        if interactions_df is None:
            # Use dataset's interactions_df if available
            interactions_df = getattr(dataset, 'interactions_df', None)
            
        if interactions_df is None:
            raise ValueError("interactions_df is required for HardNegativeSampler")
        
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # Compute item-item similarity (cosine similarity)
        item_features = user_item_matrix.T.values  # Items as rows
        self.item_similarity_matrix = cosine_similarity(item_features)
        
        # Store item mapping
        self.item_ids = user_item_matrix.columns.tolist()
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        self.is_fitted = True
        return self
    
    def sample(self, positive_items: Set[int], user_id: int = None) -> list:
        """Sample items similar to positive items as hard negatives."""
        self._check_fitted()
        
        # Get all possible negative items
        all_items = set(range(self.item_offset, self.num_items + self.item_offset))
        negative_candidates = list(all_items - positive_items)
        
        if len(negative_candidates) == 0:
            return []
        
        # Find items similar to positive items
        similar_items = set()
        
        for pos_item in positive_items:
            if pos_item in self.item_id_to_idx:
                pos_idx = self.item_id_to_idx[pos_item]
                
                # Get similarity scores for this positive item
                similarities = self.item_similarity_matrix[pos_idx]
                
                # Find items above similarity threshold
                similar_indices = np.where(similarities > self.similarity_threshold)[0]
                similar_item_ids = [self.item_ids[idx] for idx in similar_indices]
                similar_items.update(similar_item_ids)
        
        # Filter to only negative candidates that are similar
        hard_negatives = list(similar_items.intersection(negative_candidates))
        
        # If not enough hard negatives, fill with random negatives
        if len(hard_negatives) < self.num_negatives:
            remaining_negatives = list(set(negative_candidates) - similar_items)
            np.random.shuffle(remaining_negatives)
            hard_negatives.extend(remaining_negatives[:self.num_negatives - len(hard_negatives)])
        
        # Sample the requested number
        num_to_sample = min(self.num_negatives, len(hard_negatives))
        sampled_negatives = np.random.choice(hard_negatives, size=num_to_sample, replace=False)
        
        return sampled_negatives.tolist()