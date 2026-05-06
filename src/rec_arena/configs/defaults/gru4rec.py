"""GRU4Rec model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class GRU4RecConfig(BaseModelConfig):
    """GRU4Rec configuration with defaults from original paper."""

    # Architecture
    embedding_dim: int = 64
    hidden_size: int = 64
    num_layers: int = 1
    dropout_rate: float = 0.2
    max_seq_length: int = 200
    vocab_size: int = None

    # Training
    lr: float = 1e-04
    weight_decay: float = 1e-06

    # Validation metrics (optional)
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Loss function
    loss_type: str = "sampled_softmax"  # bpr, bce, gbce

    # GRU settings
    activation: str = "tanh"  # tanh, relu, gelu
    bidirectional: bool = False

    # Initialization
    init_std: float = 0.1
    
    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None

    def __post_init__(self):
        """Initialize defaults and validate configuration."""
        # Initialize defaults
        if self.val_k_values is None:
            self.val_k_values = [10]
        
        # Validate
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Set default embedding configs
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}
        
        if self.position_config is None:
            self.position_config = {"type": "learnable"}

        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
