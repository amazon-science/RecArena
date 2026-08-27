"""SimpleX model configuration."""

from dataclasses import dataclass
from typing import Optional
from ..base import BaseModelConfig


@dataclass
class SimpleXConfig(BaseModelConfig):
    """SimpleX configuration."""

    # Architecture (CCL + history aggregation, per the SimpleX paper).
    # Defaults match RecBole's SimpleX.yaml: the negative term (weighted by
    # negative_weight, filtered by margin) is SimpleX's core contribution, so a
    # small negative_weight neutralizes the model. RecBole/paper use ~10.
    embedding_dim: int = 64
    margin: float = 0.5           # CCL negative margin (RecBole default 0.5)
    negative_weight: float = 10.0  # CCL negative-loss weight w (RecBole default 10)
    gamma: float = 0.5           # user-vs-history mix: g*user + (1-g)*UI_map(agg)
    aggregator: str = "mean"     # mean | user_attention | self_attention
    dropout_prob: float = 0.1    # RecBole default 0.1
    reg_weight: float = 1e-5     # L2 EmbLoss weight (RecBole default 1e-5)

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-6  # homogeneous neural-baseline weight decay

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = None

    # SimpleX implements its own Cosine-Contrastive Loss in compute_loss and
    # does NOT use self.loss_fn. loss_type is kept as a valid factory value only
    # so the base DeepModel.__init__ (which eagerly builds a loss_fn) doesn't
    # error; it is never invoked for SimpleX.
    loss_type: str = "bpr"

    # Model settings
    num_users: Optional[int] = None
    num_items: Optional[int] = None

    # Embedding configuration
    user_embedding_config: dict = None
    item_embedding_config: dict = None

    def __post_init__(self):
        """Validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        if self.aggregator not in ["mean", "user_attention", "self_attention"]:
            raise ValueError(
                "aggregator must be mean, user_attention, or self_attention"
            )
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
