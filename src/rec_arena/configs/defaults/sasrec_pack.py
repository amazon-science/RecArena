"""SASRecPack model configuration."""

from dataclasses import dataclass, field
from ..base import BaseModelConfig


@dataclass
class SASRecPackConfig(BaseModelConfig):
    """SASRecPack: Hyperparameter-ensemble SASRec using the TabPack pattern.

    Trains multiple SASRec variants simultaneously with shared item embeddings
    but diverse architectural hyperparameters (depth, FFN dim, dropout, lr, wd).
    A greedy online ensemble selects the best subset during training.
    """

    # Shared architecture (fixed across all pack members)
    embedding_dim: int = 128
    num_heads: int = 2
    max_seq_length: int = 200
    vocab_size: int = None
    scale_embeddings: bool = True
    activation: str = "gelu"

    # Pack settings
    pack_size: int = 16

    # Per-member hyperparameter ranges (sampled uniformly)
    # Depth
    min_num_layers: int = 1
    max_num_layers: int = 4
    # FFN hidden dim (as multiplier of embedding_dim)
    min_ffn_multiplier: float = 1.0
    max_ffn_multiplier: float = 4.0
    # Dropout
    min_dropout: float = 0.0
    max_dropout: float = 0.5
    # Learning rate (log-uniform)
    min_lr: float = 1e-4
    max_lr: float = 5e-3
    # Weight decay (log-uniform)
    min_weight_decay: float = 1e-6
    max_weight_decay: float = 1e-2

    # Training
    lr: float = 1e-3  # Default (used for shared embedding LR)
    weight_decay: float = 1e-6
    batch_size: int = 256
    max_epochs: int = 1000
    patience: int = 50  # Per-member early stopping patience

    # Ensemble
    ensemble_patience: int = 32  # Stop ensemble selection after no improvement
    max_ensemble_size: int = 16  # Maximum members in final ensemble

    # Loss
    loss_type: str = "sampled_softmax"
    temperature: float = 0.1

    # Validation
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10

    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.val_k_values is None:
            self.val_k_values = [10]
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}
        if self.position_config is None:
            self.position_config = {"type": "learnable"}
        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for PAD, UNK, MASK), got {self.vocab_size}"
            )
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.pack_size < 2:
            raise ValueError("pack_size must be at least 2")
        if self.min_num_layers < 1 or self.max_num_layers < self.min_num_layers:
            raise ValueError("Invalid num_layers range")
