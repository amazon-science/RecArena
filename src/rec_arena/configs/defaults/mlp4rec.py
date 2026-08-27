"""MLP4Rec model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class MLP4RecConfig(BaseModelConfig):
    """MLP4Rec configuration - simple MLP baseline for sequential recommendation."""

    # Architecture
    embedding_dim: int = 64
    hidden_dims: list = None  # MLP hidden dimensions, e.g., [256, 128]
    num_layers: int = 2  # Number of MLP layers (if hidden_dims not specified)
    hidden_multiplier: int = 4  # Hidden dim = embedding_dim * multiplier
    dropout_rate: float = 0.2
    max_seq_length: int = 200
    vocab_size: int = None

    # Ensemble
    ensemble_size: int = 1  # 1 = no ensemble, >1 = batch ensemble

    # Pooling strategy
    pooling: str = "last"  # mean, max, last, attention, multi

    # Training
    lr: float = 1e-03
    weight_decay: float = 1e-06  # unified neural-baseline weight decay

    # Loss function
    loss_type: str = "cross_entropy"

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Architecture details
    activation: str = "relu"  # relu, gelu, swish, silu
    use_batch_norm: bool = False
    use_layer_norm: bool = True
    use_residual: bool = True  # Residual connections in MLP
    init_std: float = 0.02

    def __post_init__(self):
        """Initialize defaults and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        # Auto-generate hidden_dims if not specified
        if self.hidden_dims is None:
            hidden_dim = self.embedding_dim * self.hidden_multiplier
            self.hidden_dims = [hidden_dim] * self.num_layers

        # Allow empty hidden_dims for EmbeddingOnly baseline
        # if len(self.hidden_dims) == 0:
        #     raise ValueError("hidden_dims must have at least one dimension")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        if self.pooling not in ["mean", "max", "last", "attention", "multi"]:
            raise ValueError(
                f"pooling must be one of [mean, max, last, attention, multi], got {self.pooling}"
            )

        if self.vocab_size is not None and self.vocab_size <= 1:
            raise ValueError(f"vocab_size must be > 1, got {self.vocab_size}")

        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
