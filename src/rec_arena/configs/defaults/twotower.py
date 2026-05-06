"""Two-Tower model configuration."""

from dataclasses import dataclass, field
from typing import List, Optional
from ..base import BaseModelConfig


@dataclass
class TwoTowerConfig(BaseModelConfig):
    """Two-Tower model configuration."""

    # Architecture
    embedding_dim: int = 64
    user_tower_dims: List[int] = field(default_factory=lambda: [128, 64])
    item_tower_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.2

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = None

    # Loss function
    loss_type: str = "bpr"  # bpr, bce

    # Architecture details
    activation: str = "relu"  # relu, gelu, swish
    use_batch_norm: bool = False
    init_std: float = 0.01

    # Model parameters
    num_users: Optional[int] = None
    num_items: Optional[int] = None
    
    # Embedding configuration
    user_embedding_config: dict = None
    item_embedding_config: dict = None

    def __post_init__(self):
        """Validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if not self.user_tower_dims or not self.item_tower_dims:
            raise ValueError("Tower dimensions cannot be empty")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Set default embedding configs
        if self.user_embedding_config is None:
            self.user_embedding_config = {"type": "standard"}
        
        if self.item_embedding_config is None:
            self.item_embedding_config = {"type": "standard"}
