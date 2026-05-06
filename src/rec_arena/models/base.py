from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch


class BaseModel(ABC):
    """Abstract base class for all recommendation models."""

    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError("Config cannot be None")
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def fit(self, train_data, val_data=None) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Generate predictions for user-item pairs."""
        pass

    @abstractmethod
    def recommend(
        self, user_ids: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate top-k recommendations for users.

        Returns:
            Tuple of (recommended_items, scores)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.copy()

    def __repr__(self) -> str:
        return f"{self.name}({self.config})"
