"""SLIM model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class SLIMConfig(BaseModelConfig):
    """SLIM configuration with defaults."""

    # Model parameters (SLIMElastic literature defaults, e.g. RecBole).
    num_users: int = None
    num_items: int = None
    alpha: float = 0.2  # ElasticNet overall regularization strength
    l1_ratio: float = 0.02  # L1 vs L2 mix (mostly L2 with light sparsity)

    def __post_init__(self):
        """Validate configuration."""
        if self.num_users is None or self.num_users <= 0:
            raise ValueError("num_users must be positive")
        if self.num_items is None or self.num_items <= 0:
            raise ValueError("num_items must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")
