"""BPRMF model configuration."""

from dataclasses import dataclass
from typing import Optional
from ..base import BaseModelConfig


@dataclass
class BPRMFConfig(BaseModelConfig):
    """BPR Matrix Factorization configuration."""

    # Architecture
    embedding_dim: int = 64

    # Training. Homogeneous neural-baseline lr=1e-3. BPR-MF regularizes the
    # user+positive-item embeddings inside the loss (reg_weight) rather than via
    # optimizer weight_decay, which is the standard BPR formulation.
    reg_weight: float = 0.0001  # L2 regularization in BPR loss (only user+pos_item)
    lr: float = 1e-03
    weight_decay: float = 0.0  # in-loss reg_weight is used instead

    # Loss function
    loss_type: str = "bpr"

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = None

    # Model settings - init_std now ignored (using Xavier initialization)
    init_std: float = 0.01  # Reduced from 0.1, but Xavier init is used instead
    normalize_embeddings: bool = False
    use_bias: bool = False

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

        if self.reg_weight < 0:
            raise ValueError("reg_weight must be non-negative")

        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
