"""NCF model configuration."""

from dataclasses import dataclass, field
from typing import List, Optional
from ..base import BaseModelConfig


@dataclass
class NCFConfig(BaseModelConfig):
    """Neural Collaborative Filtering configuration."""

    # Architecture
    embedding_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2

    # Training
    lr: float = 1e-04
    weight_decay: float = 1e-06

    # Validation metrics (optional)
    compute_val_metrics: bool = False
    val_k_values: list = None

    # Loss function
    loss_type: str = "cross_entropy"  # cross_entropy, bce, sampled_softmax, bpr

    # Architecture details
    activation: str = "relu"  # relu, gelu, swish, tanh
    use_batch_norm: bool = False
    init_std: float = 0.1

    # Set during training
    num_users: Optional[int] = None
    num_items: Optional[int] = None
    
    # Embedding configuration
    user_embedding_config: dict = None
    item_embedding_config: dict = None

    def __post_init__(self):
        """Validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")

        if any(dim < 1 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Set default embedding configs
        if self.user_embedding_config is None:
            self.user_embedding_config = {"type": "standard"}
        
        if self.item_embedding_config is None:
            self.item_embedding_config = {"type": "standard"}
