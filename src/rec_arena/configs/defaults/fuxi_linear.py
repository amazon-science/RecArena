"""FuXi-Linear model configuration.

Dense-faithful port of the official FuXi-Linear (chunkwise linear-attention
retention). See ``models/sequential_models/fuxi_linear.py`` for the equivalence
argument (dense quadratic-with-decay == reference chunkwise_forward).
"""

from dataclasses import dataclass, field
from ..base import BaseModelConfig


@dataclass
class FuXiLinearConfig(BaseModelConfig):
    """FuXi-Linear configuration with defaults, mirroring FuXiGammaConfig."""

    # Architecture
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    attention_dim: int = 32
    linear_dim: int = 32  # value_dim = linear_dim * num_heads == embedding_dim
    dropout_rate: float = 0.2  # aligned with SASRec for a fair baseline
    max_seq_length: int = 200
    vocab_size: int = None

    # FFN
    ffn_multiply: float = 4.0
    epsilon: float = 1e-6
    linear_activation: str = "silu"

    # Retention (latent) channel
    use_rope: bool = False

    # Channel toggles
    enable_temporal_channel: bool = True
    enable_positional_channel: bool = True

    # Temporal channel (KRAB) parameters
    channel_t_num_heads: int = None  # defaults to num_heads
    channel_t_base: float = 2.0
    channel_t_start_index: int = 0
    channel_t_base_stride: int = 1
    channel_t_use_proj: bool = True
    channel_t_learnable_gamma: bool = False
    channel_t_no_temporal_qk: bool = False
    channel_t_aug_current: bool = False

    # Positional channel parameters
    channel_p_dim: int = 32
    channel_p_aug_current: bool = True
    channel_p_use_proj: bool = True

    # Training
    lr: float = 1e-03
    weight_decay: float = 1e-06  # unified neural-baseline weight decay

    # Loss function
    loss_type: str = "cross_entropy"

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = field(default_factory=lambda: [10])
    metric_compute_interval: int = 10  # Compute metrics every N epochs

    # Embedding configuration
    embedding_config: dict = field(default_factory=lambda: {"type": "standard"})
    position_config: dict = field(default_factory=lambda: {"type": "learnable"})

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # The reference layer-norms each channel output over embedding_dim and
        # feeds channels the (embedding_dim-wide) residual stream through a
        # value_dim projection, so value_dim = linear_dim * num_heads must equal
        # embedding_dim.
        if self.linear_dim * self.num_heads != self.embedding_dim:
            raise ValueError(
                f"linear_dim * num_heads ({self.linear_dim} * {self.num_heads} = "
                f"{self.linear_dim * self.num_heads}) must equal embedding_dim "
                f"({self.embedding_dim}); the per-channel value width is normed "
                f"over embedding_dim."
            )

        # The temporal channel reshapes the value into 2 * channel_t_num_heads
        # heads, so embedding_dim must be divisible by 2 * channel_t_num_heads.
        if self.enable_temporal_channel:
            ct_heads = self.channel_t_num_heads or self.num_heads
            if self.embedding_dim % (2 * ct_heads) != 0:
                raise ValueError(
                    f"embedding_dim ({self.embedding_dim}) must be divisible by "
                    f"2 * channel_t_num_heads (2 * {ct_heads}); the temporal "
                    f"channel splits the value into 2*num_heads heads."
                )

        # The positional channel's sinusoidal embedding dim must be even.
        if self.enable_positional_channel and self.channel_p_dim % 2 != 0:
            raise ValueError(
                f"channel_p_dim ({self.channel_p_dim}) must be even (sin/cos halves)."
            )

        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for special tokens), "
                f"got {self.vocab_size}"
            )
