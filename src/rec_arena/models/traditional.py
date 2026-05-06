import torch
import numpy as np
from typing import Dict, Any, Tuple
from .base import BaseModel


class TraditionalModel(BaseModel):
    """Base class for traditional (non-deep learning) recommendation models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_users = config.get("num_users")
        self.num_items = config.get("num_items")
        self.model = None

    def fit(self, train_data, val_data=None) -> None:
        """Train the traditional model."""
        raise NotImplementedError

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Generate predictions for user-item pairs."""
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.cpu().numpy()
        if isinstance(item_ids, torch.Tensor):
            item_ids = item_ids.cpu().numpy()

        predictions = self._predict_numpy(user_ids, item_ids)
        return torch.from_numpy(predictions).float()

    def recommend(
        self, user_ids: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate top-k recommendations for users."""
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.cpu().numpy()

        items, scores = self._recommend_numpy(user_ids, k)
        return torch.from_numpy(items), torch.from_numpy(scores)

    def _predict_numpy(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Numpy-based prediction implementation."""
        raise NotImplementedError

    def _recommend_numpy(
        self, user_ids: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numpy-based recommendation implementation."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save model using pickle or joblib."""
        import joblib
        import os
        from pathlib import Path

        try:
            # Secure path validation
            path_obj = Path(path).resolve()
            if not str(path_obj).startswith(os.getcwd()):
                raise ValueError("Path must be within current directory")
            if ".." in str(path_obj) or not path.endswith(('.pkl', '.joblib')):
                raise ValueError("Invalid path or file extension")

            os.makedirs(path_obj.parent, exist_ok=True)
            joblib.dump(self.model, path_obj)
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")

    def load(self, path: str) -> None:
        """Load model using pickle or joblib."""
        import joblib
        import os
        from pathlib import Path

        try:
            # Secure path validation
            path_obj = Path(path).resolve()
            if not str(path_obj).startswith(os.getcwd()):
                raise ValueError("Path must be within current directory")
            if not path_obj.exists() or not path.endswith(('.pkl', '.joblib')):
                raise ValueError("Invalid or non-existent path")

            self.model = joblib.load(path_obj)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
