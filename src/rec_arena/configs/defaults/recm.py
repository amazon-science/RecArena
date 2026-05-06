"""RecM model configuration."""

from dataclasses import dataclass
from typing import Optional, List
from ..base import BaseModelConfig


@dataclass
class RecMConfig(BaseModelConfig):
    """RecM configuration with ensemble defaults."""

    # Architecture
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout_rate: float = 0.1
    max_seq_length: int = 200
    vocab_size: int = None

    # Ensemble settings
    ensemble_size: int = 4
    ensemble_loss_functions: Optional[List[str]] = None  # e.g., ["bpr", "bce", "gbce"]

    # Training
    lr: float = 1e-04
    weight_decay: float = 1e-06

    # Loss function
    loss_type: str = "cross_entropy"

    # Validation metrics (optional)
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Architecture details
    transformer_activation: str = "gelu"
    use_ligr: bool = False  # Use gated residuals + SwiGLU (LiGR style)
    reuse_item_embeddings: bool = True
    scaling_init: str = "random-signs"  # Batch ensemble init: "ones", "random-signs", "normal"

    
    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1")

        # Validate ensemble loss configuration
        if self.ensemble_loss_functions is not None:
            if len(self.ensemble_loss_functions) == 1:
                # Single loss function - use for all ensemble members
                pass
            elif len(self.ensemble_loss_functions) == self.ensemble_size:
                # One loss per ensemble member - perfect
                pass
            elif self.ensemble_size % len(self.ensemble_loss_functions) == 0:
                # Divisible - distribute evenly
                pass
            else:
                raise ValueError(
                    f"ensemble_loss_functions length ({len(self.ensemble_loss_functions)}) "
                    f"must be 1, equal to ensemble_size ({self.ensemble_size}), "
                    f"or evenly divisible into ensemble_size"
                )
