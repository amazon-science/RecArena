"""Category-aware negative sampler."""

import numpy as np
import pandas as pd
from typing import Set, Dict
from .base import BaseSampler, DEFAULT_ITEM_OFFSET


class CategorySampler(BaseSampler):
    """Samples negatives from same categories as positive items."""
    
    def __init__(self, num_items: int, num_negatives: int = 50, seed: int = 42, 
                 category_prob: float = 0.7, item_offset: int = DEFAULT_ITEM_OFFSET):
        super().__init__(num_items, num_negatives, seed, item_offset)
        self.category_prob = category_prob  # Probability of sampling from same category
        self.item_categories = {}
        self.category_items = {}
        self._requires_fitting = True
        
    def fit(self, dataset, interactions_df=None, item_categories=None):
        """Fit the sampler with item category information."""
        if item_categories is None:
            # Try to get categories from dataset
            if hasattr(dataset, 'get_item_categories'):
                item_categories = dataset.get_item_categories()
            else:
                # For MovieLens, try to load genre information
                try:
                    import os
                    genre_file = os.path.join(dataset.data_path, 'u.item')
                    if os.path.exists(genre_file):
                        item_categories = self._load_movielens_genres(genre_file)
                    else:
                        raise ValueError("No category information available")
                except:
                    raise ValueError("item_categories required for CategorySampler")
        
        self.item_categories = item_categories
        
        # Create reverse mapping: category -> items
        self.category_items = {}
        for item_id, categories in item_categories.items():
            if isinstance(categories, str):
                categories = [categories]
            for category in categories:
                if category not in self.category_items:
                    self.category_items[category] = []
                self.category_items[category].append(item_id)
        
        self.is_fitted = True
        return self
    
    def _load_movielens_genres(self, genre_file):
        """Load MovieLens genre information from u.item file."""
        genres = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        item_categories = {}
        with open(genre_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 24:  # Ensure we have genre columns
                    item_id = int(parts[0])
                    item_genres = []
                    
                    # Check genre columns (last 19 columns)
                    for i, genre in enumerate(genres):
                        if i + 5 < len(parts) and parts[i + 5] == '1':
                            item_genres.append(genre)
                    
                    if item_genres:
                        item_categories[item_id] = item_genres
        
        return item_categories
    
    def sample(self, positive_items: Set[int], user_id: int = None) -> list:
        """Sample negatives preferring same categories as positive items."""
        self._check_fitted()
        
        # Get all possible negative items
        all_items = set(range(self.item_offset, self.num_items + self.item_offset))
        negative_candidates = list(all_items - positive_items)
        
        if len(negative_candidates) == 0:
            return []
        
        # Get categories of positive items
        positive_categories = set()
        for item in positive_items:
            if item in self.item_categories:
                categories = self.item_categories[item]
                if isinstance(categories, str):
                    categories = [categories]
                positive_categories.update(categories)
        
        # Sample negatives
        sampled_negatives = []
        
        for _ in range(self.num_negatives):
            if len(negative_candidates) == 0:
                break
                
            # Decide whether to sample from same category or random
            if positive_categories and np.random.random() < self.category_prob:
                # Sample from same categories
                category_candidates = []
                for category in positive_categories:
                    if category in self.category_items:
                        category_candidates.extend(self.category_items[category])
                
                # Filter to negative candidates
                category_negatives = list(set(category_candidates).intersection(negative_candidates))
                
                if category_negatives:
                    sampled_item = np.random.choice(category_negatives)
                else:
                    sampled_item = np.random.choice(negative_candidates)
            else:
                # Random sampling
                sampled_item = np.random.choice(negative_candidates)
            
            sampled_negatives.append(sampled_item)
            negative_candidates.remove(sampled_item)
        
        return sampled_negatives