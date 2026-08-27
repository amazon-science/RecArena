"""Base configuration classes."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseModelConfig:
    """Base configuration for all models."""

    embedding_dim: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0
    init_std: float = 0.1

    # Extra keyword args forwarded to the loss factory (e.g. gBCE alpha/t,
    # sampled-softmax temperature). Empty by default -> losses self-configure
    # (notably gBCE auto-calibrates alpha = num_negatives / (vocab_size - 1)).
    loss_kwargs: dict = field(default_factory=dict)
    dynamic_negative_sampling: bool = True
    early_stopping: bool = True

    # Number of negatives used by sampled losses. Populated by the benchmark
    # harness (core.py) from the resolved ModelSpec so losses that need the
    # sampling rate -- notably gBCE, whose alpha calibration = neg/(vocab-1) --
    # can self-configure. 0 means full-softmax / not applicable.
    num_negatives: int = 0

    # Validation ranking-metric computation. The harness monitors val_ndcg@10
    # (not val_loss) for early stopping + checkpoint selection, so models must
    # compute it. metric_compute_interval controls how often the (expensive)
    # full-vocab val NDCG is computed; the harness sets it to 1 so the monitored
    # metric is fresh every epoch (subclasses may still override these).
    compute_val_metrics: bool = False
    metric_compute_interval: int = 10

    # Training stability
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"  # "norm" or "value"

    # Sparse embedding tables (single-GPU memory/throughput win for large
    # catalogs). Opt-in: only safe for sampled losses (sampled_softmax, bce,
    # gbce, bpr) and plain tied/standard tables. Models auto-fall back to dense
    # (with a warning) when the configured loss/output is incompatible.
    sparse_embeddings: bool = False

    # Training loop control (consumed by the benchmark harness + configure_optimizers)
    max_epochs: int = 100
    patience: int = 10

    # Learning rate scheduler. `scheduler` is the structured dict consumed by
    # DeepModel/DeepSequentialModel.configure_optimizers (keys: type, monitor,
    # mode, patience, factor, warmup_steps). None disables scheduling.
    scheduler: Optional[dict] = None
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
