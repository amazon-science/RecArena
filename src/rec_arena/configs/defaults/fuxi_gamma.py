"""FuXi-γ model configuration."""

from dataclasses import dataclass, field
from ..base import BaseModelConfig


@dataclass
class FuXiGammaConfig(BaseModelConfig):
    """FuXi-γ configuration with exponential-power temporal encoder."""

    # Architecture
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    attention_dim: int = 64
    linear_dim: int = 64
    dropout_rate: float = 0.2  # aligned with SASRec for a fair baseline
    max_seq_length: int = 200
    vocab_size: int = None

    # FFN
    ffn_multiply: float = 4.0
    epsilon: float = 1e-6
    linear_activation: str = "silu"

    # Temporal encoder parameters
    range_alpha: float = 0.1
    left_beta: float = 0.5
    right_beta: float = 2.0
    gamma_learnable: bool = True
    left_gamma: float = 0.5
    right_gamma: float = 0.99

    # Training
    lr: float = 1e-03
    weight_decay: float = 1e-06  # unified neural-baseline weight decay

    # Loss function
    loss_type: str = "cross_entropy"

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = field(default_factory=lambda: [10])
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Embedding configuration
    embedding_config: dict = field(default_factory=lambda: {"type": "standard"})
    position_config: dict = field(default_factory=lambda: {"type": "learnable"})

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for special tokens), got {self.vocab_size}"
            )
