"""EASE model configuration."""

from dataclasses import dataclass
from typing import Optional
from ..base import BaseModelConfig


@dataclass
class EASEConfig(BaseModelConfig):
    """EASE configuration with defaults."""

    # Model parameters
    num_users: int = None
    num_items: int = None
    # Regularization strength (lambda in P = X^T X + lambda*I). ``reg_weight`` is
    # the name RecBole and the comparison harness use; ``reg_lambda`` is kept for
    # backward compat. If ``reg_weight`` is provided it takes precedence and
    # overwrites ``reg_lambda`` so the model reads a single canonical value.
    reg_lambda: float = 500.0  # Regularization parameter (typical: 100-1000)
    reg_weight: Optional[float] = None  # alias for reg_lambda (RecBole naming)

    def __post_init__(self):
        """Validate configuration."""
        if self.num_users is None or self.num_users <= 0:
            raise ValueError("num_users must be positive")
        if self.num_items is None or self.num_items <= 0:
            raise ValueError("num_items must be positive")
        # reg_weight is an alias for reg_lambda; when set it wins.
        if self.reg_weight is not None:
            self.reg_lambda = self.reg_weight
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
