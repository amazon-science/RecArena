"""HSTU model configuration."""

from dataclasses import dataclass, field
from ..base import BaseModelConfig


@dataclass
class HSTUConfig(BaseModelConfig):
    """HSTU configuration with defaults (HSTU-Large from paper)."""

    # Architecture (HSTU-Large defaults)
    embedding_dim: int = 50
    num_heads: int = 2
    num_layers: int = 8
    dropout_rate: float = 0.2
    max_seq_length: int = 200
    vocab_size: int = None

    # Training
    lr: float = 1e-03
    weight_decay: float = 1e-07

    # Loss function
    loss_type: str = "sampled_softmax"

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = field(default_factory=lambda: [10])
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Embedding configuration
    embedding_config: dict = field(default_factory=lambda: {"type": "standard"})
    position_config: dict = field(default_factory=lambda: {"type": "learnable"})

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for PAD, UNK, MASK), got {self.vocab_size}"
            )
