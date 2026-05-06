"""ItemKNN model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class ItemKNNConfig(BaseModelConfig):
    """ItemKNN configuration with defaults."""

    # Model parameters
    num_users: int = None
    num_items: int = None
    k: int = 100  # Number of neighbors
    similarity: str = "cosine"  # 'cosine' or 'jaccard'
    shrinkage: float = 100.0  # Shrinkage parameter for cosine similarity
    normalize: bool = True  # Normalize similarity scores

    def __post_init__(self):
        """Validate configuration."""
        if self.num_users is None or self.num_users <= 0:
            raise ValueError("num_users must be positive")
        if self.num_items is None or self.num_items <= 0:
            raise ValueError("num_items must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.similarity not in ["cosine", "jaccard"]:
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        if self.shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
