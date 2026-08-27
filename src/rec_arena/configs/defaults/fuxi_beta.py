"""FuXi-β model configuration."""

from dataclasses import dataclass, field
from ..base import BaseModelConfig


@dataclass
class FuXiBetaConfig(BaseModelConfig):
    """FuXi-β configuration with functional relative attention bias.

    Dense-faithful port of the reference ``fuxi_beta`` block. The temporal bias
    ``f()`` is selectable via ``func_type`` (linear/log/exp/sin/pow/mixed/nn/zero,
    default 'pow' = a*(relu(Δ)+1)^(-b)) and complemented by a learnable
    positional Toeplitz bias.
    """

    # Architecture
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    # Value/attention hidden width per head. The internal projection is sized
    # linear_dim * num_heads * 3 (u = 2 chunks, v = 1 chunk), so the model
    # hidden width is decoupled from embedding_dim.
    linear_dim: int = 64
    attention_dim: int = 64
    dropout_rate: float = 0.2  # aligned with SASRec for a fair baseline
    max_seq_length: int = 200
    vocab_size: int = None

    # FFN (MultistageFeedforwardNeuralNetwork)
    ffn_multiply: float = 4.0
    epsilon: float = 1e-6
    linear_activation: str = "silu"  # activation applied over the u/v projection

    # Temporal bias function selector.
    func_type: str = "pow"

    # ---- Function-specific initialization ranges (low, high) for the learnable
    # temporal-bias parameters. Mirrors the reference uniform inits. Only the
    # ranges for the active ``func_type`` instantiate parameters; the rest are
    # passed through but stay inert (see FunctionalRelativeAttentionBias).
    lin_a_range: tuple = (-0.01, 0.01)  # linear: a*x + b
    lin_b_range: tuple = (-0.2, 0.2)
    log_a_range: tuple = (-0.01, 0.01)  # log: a*log(1 + b*relu(x)) + c
    log_b_range: tuple = (0.5, 1.0)
    log_c_range: tuple = (-0.05, 0.05)
    exp_a_range: tuple = (-0.2, 0.2)  # exp: a*exp(-exp(b)*relu(x))
    exp_b_range: tuple = (-2.0, 0.0)
    sin_a_range: tuple = (-0.02, 0.02)  # sin: c*sin(a*x + b) + d
    sin_b_range: tuple = (-3.141592653589793, 3.141592653589793)
    sin_c_range: tuple = (-2.0, 2.0)
    sin_d_init: float = 0.0
    pow_a_range: tuple = (-0.2, 0.2)  # pow: a*(relu(x)+1)^(-b)
    pow_b_range: tuple = (0.4, 0.8)
    # nn func: single hidden width for the 3-layer MLP temporal encoder.
    nn_hidden_dim: int = 5

    # Training
    lr: float = 1e-03
    weight_decay: float = 1e-06  # unified neural-baseline weight decay

    # Loss function
    loss_type: str = "cross_entropy"

    # Validation metrics
    compute_val_metrics: bool = False
    val_k_values: list = field(default_factory=lambda: [10])
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Embedding configuration
    embedding_config: dict = field(default_factory=lambda: {"type": "standard"})

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        valid_funcs = {"linear", "log", "exp", "sin", "pow", "mixed", "nn", "zero"}
        if self.func_type not in valid_funcs:
            raise ValueError(
                f"func_type must be one of {sorted(valid_funcs)}, got "
                f"'{self.func_type}'"
            )

        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for special tokens), got "
                f"{self.vocab_size}"
            )
