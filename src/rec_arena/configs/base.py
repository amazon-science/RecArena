"""Base configuration classes."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseModelConfig:
    """Base configuration for all models."""

    embedding_dim: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0
    init_std: float = 0.1
    dynamic_negative_sampling: bool = True
    early_stopping: bool = True

    # Training stability
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"  # "norm" or "value"

    # Learning rate scheduler
    use_scheduler: bool = False
    scheduler_type: str = "cosine"  # cosine, reduce_on_plateau, step
    warmup_steps: int = 0
    scheduler_patience: int = 5  # For reduce_on_plateau
    scheduler_factor: float = 0.5  # For reduce_on_plateau/step

    def get(self, key: str, default=None):
        """Get attribute value with default fallback."""
        return getattr(self, key, default)


@dataclass
class BaseTrainingConfig:
    """Base training configuration."""

    max_epochs: int = 100
    batch_size: int = 256
    patience: int = 10
    gradient_clip_val: float = 1.0
    precision: str = "32"
    early_stopping: bool = True


@dataclass
class BaseDataConfig:
    """Base data configuration."""

    name: str = "ml100k"
    min_interactions: int = 5
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    implicit_threshold: float = 4.0
