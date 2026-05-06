"""EASE model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class EASEConfig(BaseModelConfig):
    """EASE configuration with defaults."""

    # Model parameters
    num_users: int = None
    num_items: int = None
    reg_lambda: float = 500.0  # Regularization parameter (typical: 100-1000)

    def __post_init__(self):
        """Validate configuration."""
        if self.num_users is None or self.num_users <= 0:
            raise ValueError("num_users must be positive")
        if self.num_items is None or self.num_items <= 0:
            raise ValueError("num_items must be positive")
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
