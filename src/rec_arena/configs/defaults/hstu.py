"""HSTU model configuration."""

from dataclasses import dataclass, field
from ..base import BaseModelConfig


@dataclass
class HSTUConfig(BaseModelConfig):
    """HSTU configuration with defaults (HSTU-Large from paper)."""

    # Architecture (HSTU-Large defaults)
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 4  # HSTU has no FFN, so needs more layers for equivalent capacity
    dropout_rate: float = 0.2
    max_seq_length: int = 200
    vocab_size: int = None

    # Decoupled per-head dims (reference dqk / dv). If None, default to
    # embedding_dim // num_heads. The ML-1M reference uses dqk = dv = 50.
    attention_dim: int = None
    linear_dim: int = None

    # Gating variant. concat_ua=False (reference default) -> o_input = u * norm(a).
    # concat_ua=True -> o_input = cat([u, norm(a), u*norm(a)]).
    concat_ua: bool = False

    # Training (homogeneous neural-baseline recipe: lr=1e-3, wd=1e-6)
    lr: float = 1e-03
    weight_decay: float = 1e-06

    # Loss function
    loss_type: str = "sampled_softmax"

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = field(default_factory=lambda: [10])
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Embedding configuration
    embedding_config: dict = field(default_factory=lambda: {"type": "standard"})
    position_config: dict = field(default_factory=lambda: {"type": "learnable"})

    # Relative time+position bias (HSTU reference). When True, uses real
    # interaction timestamps (bucketed log-Δt) in addition to relative position;
    # falls back to position-only if timestamps are unavailable at run time.
    use_time_bias: bool = True
    num_time_buckets: int = 128

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for PAD, UNK, MASK), got {self.vocab_size}"
            )
