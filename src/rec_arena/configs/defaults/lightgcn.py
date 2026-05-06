"""LightGCN model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class LightGCNConfig(BaseModelConfig):
    """LightGCN configuration with defaults from original paper."""

    # Architecture
    embedding_dim: int = 64
    num_layers: int = 3
    
    # Graph parameters
    num_users: int = None
    num_items: int = None
    
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss function
    loss_type: str = "bpr"
    
    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = None

    
    # Embedding configuration
    user_embedding_config: dict = None
    item_embedding_config: dict = None
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10, 20]
        
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")