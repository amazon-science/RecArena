"""Genre-based negative samplers for MovieLens - OPTIMIZED VERSION."""

import numpy as np
from typing import Set, Dict, List
from .base import BaseSampler, DEFAULT_ITEM_OFFSET


class GenreDiverseSampler(BaseSampler):
    """Sample negatives from DIFFERENT genres than positive items.
    
    OPTIMIZED: Precomputes genre-to-item mapping for O(1) lookups.
    """
    
    def __init__(self, num_items: int, item_genres: Dict[int, List[int]], 
                 num_negatives: int = 50, seed: int = 42, item_offset: int = DEFAULT_ITEM_OFFSET):
        super().__init__(num_items, num_negatives, seed, item_offset)
        self.item_genres = item_genres
        self.rng = np.random.default_rng(seed)
        self._requires_fitting = False
        self.is_fitted = True
        
        # OPTIMIZATION: Precompute genre -> items mapping
        self.genre_to_items: Dict[int, Set[int]] = {}
        self.all_items = set(range(item_offset, num_items + item_offset))
        self.all_items_array = np.array(list(self.all_items))
        
        for item_id, genres in item_genres.items():
            for genre in genres:
                if genre not in self.genre_to_items:
                    self.genre_to_items[genre] = set()
                self.genre_to_items[genre].add(item_id)
    
    def _get_diverse_candidates(self, positive_items: Set[int]) -> np.ndarray:
        """Get candidates with no genre overlap - OPTIMIZED."""
        # Get all genres from positive items
        positive_genres = set()
        for item_id in positive_items:
            positive_genres.update(self.item_genres.get(item_id, []))
        
        if not positive_genres:
            # No genre info, return all non-positive items
            candidates = self.all_items - positive_items
            return np.array(list(candidates)) if candidates else np.array([])
        
        # OPTIMIZATION: Use precomputed mapping to get items to exclude
        items_with_overlap = set()
        for genre in positive_genres:
            items_with_overlap.update(self.genre_to_items.get(genre, set()))
        
        # Candidates = all items - positive items - items with genre overlap
        candidates = self.all_items - positive_items - items_with_overlap
        
        if len(candidates) < self.num_negatives:
            # Fallback: use all non-positive items
            candidates = self.all_items - positive_items
        
        return np.array(list(candidates)) if candidates else np.array([])
    
    def sample(self, positive_items: Set[int], user_id: int = None) -> list:
        """Sample items with different genres."""
        candidates = self._get_diverse_candidates(positive_items)
        if len(candidates) == 0:
            return []
        num_to_sample = min(self.num_negatives, len(candidates))
        return self.rng.choice(candidates, size=num_to_sample, replace=False).tolist()
    
    def sample_many(self, positive_items: Set[int], count: int, user_id: int = None) -> np.ndarray:
        """Sample count negatives with genre diversity (with replacement)."""
        candidates = self._get_diverse_candidates(positive_items)
        if len(candidates) == 0:
            return np.zeros(count, dtype=np.int64)
        return self.rng.choice(candidates, size=count, replace=True)


class GenreSimilarSampler(BaseSampler):
    """Sample negatives from SAME genres as positive items (harder negatives).
    
    OPTIMIZED: Precomputes genre-to-item mapping for O(1) lookups.
    """
    
    def __init__(self, num_items: int, item_genres: Dict[int, List[int]], 
                 num_negatives: int = 50, seed: int = 42, item_offset: int = DEFAULT_ITEM_OFFSET):
        super().__init__(num_items, num_negatives, seed, item_offset)
        self.item_genres = item_genres
        self.rng = np.random.default_rng(seed)
        self._requires_fitting = False
        self.is_fitted = True
        
        # OPTIMIZATION: Precompute genre -> items mapping
        self.genre_to_items: Dict[int, Set[int]] = {}
        self.all_items = set(range(item_offset, num_items + item_offset))
        
        for item_id, genres in item_genres.items():
            for genre in genres:
                if genre not in self.genre_to_items:
                    self.genre_to_items[genre] = set()
                self.genre_to_items[genre].add(item_id)
        
        # OPTIMIZATION: Precompute item -> genre count for weighting
        self.item_genre_count = {item: len(genres) for item, genres in item_genres.items()}
    
    def _get_similar_candidates(self, positive_items: Set[int]):
        """Get candidates with genre overlap and their weights - OPTIMIZED."""
        # Get all genres from positive items
        positive_genres = set()
        for item_id in positive_items:
            positive_genres.update(self.item_genres.get(item_id, []))
        
        if len(positive_genres) == 0:
            candidates = np.array(list(self.all_items - positive_items))
            return candidates, None
        
        # OPTIMIZATION: Use precomputed mapping to get candidates with overlap
        candidates_with_overlap = set()
        for genre in positive_genres:
            candidates_with_overlap.update(self.genre_to_items.get(genre, set()))
        
        # Remove positive items
        candidates_with_overlap -= positive_items
        
        if len(candidates_with_overlap) < self.num_negatives:
            candidates = np.array(list(self.all_items - positive_items))
            return candidates, None
        
        # Compute overlap scores for weighting
        candidates_list = list(candidates_with_overlap)
        scores = np.zeros(len(candidates_list), dtype=np.float64)
        
        for i, item_id in enumerate(candidates_list):
            item_genres_set = set(self.item_genres.get(item_id, []))
            scores[i] = len(item_genres_set & positive_genres)
        
        probs = scores / scores.sum()
        return np.array(candidates_list), probs
    
    def sample(self, positive_items: Set[int], user_id: int = None) -> list:
        """Sample items with similar genres (harder negatives)."""
        candidates, probs = self._get_similar_candidates(positive_items)
        if len(candidates) == 0:
            return []
        num_to_sample = min(self.num_negatives, len(candidates))
        sampled = self.rng.choice(candidates, size=num_to_sample, replace=False, p=probs)
        return sampled.tolist()
    
    def sample_many(self, positive_items: Set[int], count: int, user_id: int = None) -> np.ndarray:
        """Sample count negatives with genre similarity weighting (with replacement)."""
        candidates, probs = self._get_similar_candidates(positive_items)
        if len(candidates) == 0:
            return np.zeros(count, dtype=np.int64)
        return self.rng.choice(candidates, size=count, replace=True, p=probs)
