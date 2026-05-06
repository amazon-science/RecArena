"""FMLP-Rec model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class FMLPRecConfig(BaseModelConfig):
    """FMLP-Rec (Filter-enhanced MLP) configuration."""

    # Architecture
    embedding_dim: int = 64
    num_blocks: int = 2
    mlp_hidden_dim: int = None  # Hidden dimension for MLPs (default: 4 * embedding_dim)
    dropout_rate: float = 0.5  # Original default

    # Training matching original
    lr: float = 1e-3  # Original learning rate
    weight_decay: float = 0.0  # Original uses NO weight decay
    gradient_clip_val: float = 1.0

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Loss function
    loss_type: str = "cross_entropy"

    # Model settings
    vocab_size: int = None
    max_seq_length: int = 200

    
    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None
    def __post_init__(self):
        """Validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        # Default MLP hidden dim to 4x embedding_dim (common practice)
        if self.mlp_hidden_dim is None:
            self.mlp_hidden_dim = 4 * self.embedding_dim

        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        if self.mlp_hidden_dim < 1:
            raise ValueError("mlp_hidden_dim must be positive")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        if self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive")
        
        # Set default embedding configs
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}
        
        if self.position_config is None:
            self.position_config = {"type": "learnable"}
