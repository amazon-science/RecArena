import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from ..traditional import TraditionalModel
from ...configs.defaults.itemknn import ItemKNNConfig


class ItemKNN(TraditionalModel):
    """ItemKNN: Item-based K-Nearest Neighbors Collaborative Filtering.
    
    Classic item-based collaborative filtering using cosine similarity.
    Predicts user preferences based on similar items they've interacted with.
    
    Paper: "Item-based collaborative filtering recommendation algorithms" (WWW 2001)
    Link: https://dl.acm.org/doi/10.1145/371920.372071
    
    Model ID: itemknn
    Model Type: Implicit Feedback
    
    Key Features:
        - Simple and interpretable
        - Fast inference
        - No training required (just compute similarities)
        - Works well with sparse data
    
    Args:
        config (ItemKNNConfig): Model configuration with k and similarity parameters
    
    Example:
        >>> config = ItemKNNConfig(num_users=1000, num_items=500, k=100)
        >>> model = ItemKNN(config)
        >>> model.fit(train_data)
        >>> scores = model.predict(user_ids, item_ids)
    """
    
    def __init__(self, config: ItemKNNConfig):
        super().__init__(config)
        self.k = config.k  # Number of neighbors
        self.similarity = config.similarity  # 'cosine', 'jaccard', 'pearson'
        self.shrinkage = config.shrinkage  # Shrinkage parameter
        self.normalize = config.normalize
        self.similarity_matrix = None
        self.user_item_matrix = None
    
    def fit(self, train_data, val_data=None):
        """Compute item-item similarity matrix.
        
        Args:
            train_data: Either a sparse matrix or TraditionalDataModule
        """
        # Handle different input types
        if sp.issparse(train_data):
            X = train_data
        elif hasattr(train_data, 'get_train_matrix'):
            X = train_data.get_train_matrix()
        else:
            raise ValueError(
                "train_data must be either a sparse matrix or TraditionalDataModule"
            )
        
        self.user_item_matrix = X
        
        # Compute item-item similarity
        if self.similarity == 'cosine':
            # Cosine similarity with shrinkage
            X_items = X.T.toarray()  # [num_items, num_users]
            
            if self.shrinkage > 0:
                # Shrunk cosine similarity
                norms = np.linalg.norm(X_items, axis=1, keepdims=True)
                norms_product = norms @ norms.T
                
                dot_product = X_items @ X_items.T
                
                # Apply shrinkage
                self.similarity_matrix = dot_product / (norms_product + self.shrinkage)
            else:
                # Standard cosine similarity
                self.similarity_matrix = cosine_similarity(X_items)
        
        elif self.similarity == 'jaccard':
            # Jaccard similarity
            X_items = X.T.toarray().astype(bool)
            intersection = X_items @ X_items.T
            
            row_sums = X_items.sum(axis=1, keepdims=True)
            union = row_sums + row_sums.T - intersection
            
            self.similarity_matrix = intersection / (union + 1e-10)
        
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")
        
        # Set diagonal to zero (no self-similarity) BEFORE top-k selection.
        # Otherwise each item's own self-similarity (=1.0, always the largest)
        # occupies a top-k slot that then gets zeroed, leaving only k-1 real
        # neighbors. RecBole zeroes self first (compute_similarity), keeping a
        # full k neighbors -- matching that avoids an off-by-one neighbor count.
        np.fill_diagonal(self.similarity_matrix, 0)

        # Keep only top-k neighbors, truncated PER COLUMN.
        #
        # Scoring is `user_vectors @ similarity_matrix`, i.e.
        #   score(u, j) = sum_i X[u,i] * S[i,j].
        # Standard item-based CF (Sarwar et al. 2001) scores a TARGET item j from
        # ITS k nearest neighbours in the user's history:
        #   score(u, j) = sum_{i in KNN(j)} X[u,i] * sim(i,j),
        # so column j of S must hold item j's neighbours -> top-k PER COLUMN.
        # Truncating per ROW (the previous behaviour) keeps "items for which j is
        # a neighbour", a different, non-standard quantity, and produced a large
        # ranking discrepancy vs RecBole / the literature. The pre-truncation
        # cosine matrix is symmetric, so per-column truncation matches RecBole's
        # ComputeSimilarity exactly (up to tie-breaking on equal similarities).
        if self.k < self.num_items:
            S = self.similarity_matrix
            keep = np.zeros_like(S, dtype=bool)
            for j in range(self.num_items):
                col = S[:, j]
                top_k_idx = np.argpartition(col, -self.k)[-self.k:]
                keep[top_k_idx, j] = True
            S[~keep] = 0
            self.similarity_matrix = S

        # Normalize if requested. Normalize PER COLUMN to match the per-column
        # neighbour truncation (each target item's neighbour weights sum to 1).
        #
        # NOTE: this path defaults to OFF (config normalize=False). Column-sum
        # normalization rescales each target item's column independently, which
        # is NOT rank-preserving across target items and diverges from standard
        # item-based CF (Sarwar et al. 2001) / RecBole, whose convention is
        # UNNORMALIZED neighbour weights. Kept available for explicit opt-in.
        if self.normalize:
            col_sums = self.similarity_matrix.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            self.similarity_matrix = self.similarity_matrix / col_sums
    
    def _predict_numpy(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Predict scores for user-item pairs."""
        if self.similarity_matrix is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Get user interaction vectors
        user_vectors = self.user_item_matrix[user_ids].toarray()
        
        # Compute scores: X @ S
        all_scores = user_vectors @ self.similarity_matrix
        
        # Extract scores for specific items
        scores = all_scores[np.arange(len(user_ids)), item_ids]
        
        return scores
    
    def _recommend_numpy(self, user_ids: np.ndarray, k: int) -> tuple:
        """Generate top-k recommendations for users."""
        if self.similarity_matrix is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Get user interaction vectors
        user_vectors = self.user_item_matrix[user_ids].toarray()
        
        # Compute scores for all items: X @ S
        scores = user_vectors @ self.similarity_matrix
        
        # Get top-k items
        top_items = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_items, axis=1)
        
        return top_items, top_scores
    
    def save(self, path: str):
        """Save ItemKNN model."""
        import joblib
        import os
        from pathlib import Path
        
        path_obj = Path(path).resolve()
        os.makedirs(path_obj.parent, exist_ok=True)
        
        joblib.dump({
            'similarity_matrix': self.similarity_matrix,
            'user_item_matrix': self.user_item_matrix,
            'config': self.config
        }, path_obj)
    
    def load(self, path: str):
        """Load ItemKNN model."""
        import joblib
        from pathlib import Path
        
        path_obj = Path(path).resolve()
        data = joblib.load(path_obj)
        
        self.similarity_matrix = data['similarity_matrix']
        self.user_item_matrix = data['user_item_matrix']
