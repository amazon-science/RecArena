import torch
import numpy as np
import scipy.sparse as sp
from ..traditional import TraditionalModel
from ...configs.defaults.ease import EASEConfig


class EASE(TraditionalModel):
    """EASE: Embarrassingly Shallow Autoencoders for Sparse Data.
    
    A linear autoencoder with closed-form solution that achieves strong performance
    despite its simplicity. No training iterations needed!
    
    Paper: "Embarrassingly Shallow Autoencoders for Sparse Data" (WWW 2019)
    Link: https://dl.acm.org/doi/10.1145/3308558.3313710
    
    Model ID: ease
    Model Type: Implicit Feedback
    
    Key Features:
        - Closed-form solution (no training loop)
        - Extremely fast training (seconds)
        - Often beats complex neural models
        - Linear and interpretable
        - Item-item similarity matrix
    
    Args:
        config (EASEConfig): Model configuration with regularization parameter
    
    Example:
        >>> config = EASEConfig(num_users=1000, num_items=500, reg_lambda=500)
        >>> model = EASE(config)
        >>> model.fit(train_data)
        >>> scores = model.predict(user_ids, item_ids)
    """
    
    def __init__(self, config: EASEConfig):
        super().__init__(config)
        self.reg_lambda = config.reg_lambda
        self.B = None  # Item-item similarity matrix
        self.user_item_matrix = None
    
    def fit(self, train_data, val_data=None):
        """Train EASE with closed-form solution.
        
        Solves: min ||X - XB||² + λ||B||² subject to diag(B) = 0
        Solution: B = (X^T X + λI)^(-1) - I/diag(...)
        
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
        
        # Compute Gram matrix: G = X^T X
        G = X.T @ X
        G = G.toarray()
        
        # Add regularization: G + λI
        G += self.reg_lambda * np.eye(self.num_items)
        
        # Solve: P = G^(-1)
        P = np.linalg.inv(G)
        
        # Compute B: B = I - P / diag(P)
        self.B = np.eye(self.num_items) - P / np.diag(P)
        
        # Set diagonal to zero (no self-loops)
        np.fill_diagonal(self.B, 0.0)
    
    def _predict_numpy(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Predict scores for user-item pairs."""
        if self.B is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Get user interaction vectors
        user_vectors = self.user_item_matrix[user_ids].toarray()
        
        # Compute scores: X @ B
        all_scores = user_vectors @ self.B
        
        # Extract scores for specific items
        scores = all_scores[np.arange(len(user_ids)), item_ids]
        
        return scores
    
    def _recommend_numpy(self, user_ids: np.ndarray, k: int) -> tuple:
        """Generate top-k recommendations for users."""
        if self.B is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Get user interaction vectors
        user_vectors = self.user_item_matrix[user_ids].toarray()
        
        # Compute scores for all items: X @ B
        scores = user_vectors @ self.B
        
        # Get top-k items
        top_items = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_items, axis=1)
        
        return top_items, top_scores
    
    def save(self, path: str):
        """Save EASE model."""
        import joblib
        import os
        from pathlib import Path
        
        path_obj = Path(path).resolve()
        os.makedirs(path_obj.parent, exist_ok=True)
        
        joblib.dump({
            'B': self.B,
            'user_item_matrix': self.user_item_matrix,
            'config': self.config
        }, path_obj)
    
    def load(self, path: str):
        """Load EASE model."""
        import joblib
        from pathlib import Path
        
        path_obj = Path(path).resolve()
        data = joblib.load(path_obj)
        
        self.B = data['B']
        self.user_item_matrix = data['user_item_matrix']
