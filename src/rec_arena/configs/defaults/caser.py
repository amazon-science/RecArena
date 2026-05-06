"""Caser model configuration."""

from dataclasses import dataclass, field
from typing import List
from ..base import BaseModelConfig


@dataclass
class CaserConfig(BaseModelConfig):
    """Caser (Convolutional Sequence Embedding) configuration."""

    # Architecture
    embedding_dim: int = 64
    num_horizontal_filters: int = 16
    num_vertical_filters: int = 4
    horizontal_filter_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    vertical_filter_size: int = None  # Default to max_seq_length
    activation: str = "relu"  # relu, gelu, swish, tanh
    dropout_rate: float = 0.5

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5

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
        
        # Default vertical filter size to max_seq_length
        if self.vertical_filter_size is None:
            self.vertical_filter_size = self.max_seq_length

        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Set default embedding configs
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}
        
        if self.position_config is None:
            self.position_config = {"type": "learnable"}
        
        # Validate horizontal filter sizes
        if not self.horizontal_filter_sizes:
            raise ValueError("horizontal_filter_sizes cannot be empty")
        
        if max(self.horizontal_filter_sizes) > self.max_seq_length:
            raise ValueError(
                f"Max horizontal filter size {max(self.horizontal_filter_sizes)} "
                f"exceeds max_seq_length {self.max_seq_length}"
            )
        
        if min(self.horizontal_filter_sizes) < 1:
            raise ValueError("Horizontal filter sizes must be at least 1")
        
        # Validate vertical filter size
        if self.vertical_filter_size > self.max_seq_length:
            raise ValueError(
                f"vertical_filter_size {self.vertical_filter_size} "
                f"exceeds max_seq_length {self.max_seq_length}"
            )
        
        if self.vertical_filter_size < 1:
            raise ValueError("vertical_filter_size must be at least 1")
        
        # Warn if filters are too large
        if max(self.horizontal_filter_sizes) > self.max_seq_length // 2:
            import warnings
            warnings.warn(
                f"Large horizontal filter sizes (max={max(self.horizontal_filter_sizes)}) "
                f"relative to sequence length ({self.max_seq_length}) may reduce effectiveness"
            )
