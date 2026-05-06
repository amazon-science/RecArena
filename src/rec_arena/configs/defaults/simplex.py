"""SimpleX model configuration."""

from dataclasses import dataclass
from typing import Optional
from ..base import BaseModelConfig


@dataclass
class SimpleXConfig(BaseModelConfig):
    """SimpleX configuration."""

    # Architecture
    embedding_dim: int = 64
    margin: float = 0.9
    negative_weight: float = 0.5
    history_aggregation: str = "mean"  # mean, sum, max

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = None

    # Loss function
    loss_type: str = "bpr"

    # Model settings
    num_users: Optional[int] = None
    num_items: Optional[int] = None
    
    # Embedding configuration
    user_embedding_config: dict = None
    item_embedding_config: dict = None

    
    # Embedding configuration
    user_embedding_config: dict = None
    item_embedding_config: dict = None
    def __post_init__(self):
        """Validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        if self.history_aggregation not in ["mean", "sum", "max"]:
            raise ValueError("history_aggregation must be mean, sum, or max")
