import torch
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet
from ..traditional import TraditionalModel
from ...configs.defaults.slim import SLIMConfig


class SLIM(TraditionalModel):
    """SLIM: Sparse Linear Methods for Top-N Recommender Systems.
    
    Learns a sparse item-item similarity matrix using elastic net regularization.
    Each item is predicted as a sparse linear combination of other items.
    
    Paper: "SLIM: Sparse Linear Methods for Top-N Recommender Systems" (ICDM 2011)
    Link: https://ieeexplore.ieee.org/document/6137254
    
    Model ID: slim
    Model Type: Implicit Feedback
    
    Key Features:
        - Sparse item-item similarity matrix
        - L1 + L2 regularization (elastic net)
        - Interpretable linear model
        - Strong performance on sparse data
    
    Args:
        config (SLIMConfig): Model configuration with regularization parameters
    
    Example:
        >>> config = SLIMConfig(num_users=1000, num_items=500, alpha=0.1, l1_ratio=0.1)
        >>> model = SLIM(config)
        >>> model.fit(train_data)
        >>> scores = model.predict(user_ids, item_ids)
    """
    
    def __init__(self, config: SLIMConfig):
        super().__init__(config)
        self.alpha = config.alpha  # Regularization strength
        self.l1_ratio = config.l1_ratio  # L1 vs L2 ratio
        self.W = None  # Item-item similarity matrix
        self.user_item_matrix = None
    
    def fit(self, train_data, val_data=None):
        """Train SLIM using coordinate descent with elastic net.
        
        For each item j, solve:
            min ||X_j - X_{-j} w_j||² + α(l1_ratio||w_j||₁ + (1-l1_ratio)||w_j||²)
        
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
        
        # Initialize similarity matrix
        self.W = sp.lil_matrix((self.num_items, self.num_items), dtype=np.float32)
        
        # Train elastic net for each item
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            positive=True,  # Non-negative weights
            fit_intercept=False,
            max_iter=100,
            selection='random'
        )
        
        # Learn similarity for each item
        for j in range(self.num_items):
            # Target: item j
            y = X[:, j].toarray().ravel()
            
            # Features: all other items
            start_idx = max(0, j - 1)
            X_train = sp.hstack([X[:, :j], X[:, j+1:]], format='csr')
            
            # Skip if no interactions
            if y.sum() == 0:
                continue
            
            # Fit elastic net
            model.fit(X_train, y)
            
            # Store non-zero weights
            w = model.coef_
            w_idx = np.where(w > 0)[0]
            
            # Adjust indices (account for removed column j)
            w_idx_adjusted = np.where(w_idx < j, w_idx, w_idx + 1)
            
            self.W[w_idx_adjusted, j] = w[w_idx]
        
        # Convert to CSR for efficient operations
        self.W = self.W.tocsr()
    
    def _predict_numpy(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Predict scores for user-item pairs."""
        if self.W is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Get user interaction vectors
        user_vectors = self.user_item_matrix[user_ids]
        
        # Compute scores: X @ W
        all_scores = user_vectors @ self.W
        all_scores = all_scores.toarray()
        
        # Extract scores for specific items
        scores = all_scores[np.arange(len(user_ids)), item_ids]
        
        return scores
    
    def _recommend_numpy(self, user_ids: np.ndarray, k: int) -> tuple:
        """Generate top-k recommendations for users."""
        if self.W is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Get user interaction vectors
        user_vectors = self.user_item_matrix[user_ids]
        
        # Compute scores for all items: X @ W
        scores = user_vectors @ self.W
        scores = scores.toarray()
        
        # Get top-k items
        top_items = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_items, axis=1)
        
        return top_items, top_scores
    
    def save(self, path: str):
        """Save SLIM model."""
        import joblib
        import os
        from pathlib import Path
        
        path_obj = Path(path).resolve()
        os.makedirs(path_obj.parent, exist_ok=True)
        
        joblib.dump({
            'W': self.W,
            'user_item_matrix': self.user_item_matrix,
            'config': self.config
        }, path_obj)
    
    def load(self, path: str):
        """Load SLIM model."""
        import joblib
        from pathlib import Path
        
        path_obj = Path(path).resolve()
        data = joblib.load(path_obj)
        
        self.W = data['W']
        self.user_item_matrix = data['user_item_matrix']
