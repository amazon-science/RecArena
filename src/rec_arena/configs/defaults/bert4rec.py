"""BERT4Rec model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class BERT4RecConfig(BaseModelConfig):
    """BERT4Rec configuration with defaults from original paper."""

    # Architecture
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout_rate: float = 0.1
    max_seq_length: int = 200
    vocab_size: int = None

    # Training
    lr: float = 1e-04
    weight_decay: float = 1e-06

    # Validation metrics (optional)
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Masking strategy
    mask_prob: float = 0.2  # Probability of masking a token
    mask_token_prob: float = 0.8  # 80% replace with [MASK]
    random_token_prob: float = 0.1  # 10% replace with random token
    # Remaining 10% keep original token

    # Loss function
    loss_type: str = "cross_entropy"
    transformer_activation: str = "gelu"
    label_smoothing: float = 0.1
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    # Architecture details
    activation: str = "gelu"  # gelu, relu, swish
    use_ligr: bool = False  # Use gated residuals + SwiGLU (LiGR style)
    layer_norm_first: bool = True
    init_std: float = 0.02
    use_bias: bool = True
    
    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None

    def __post_init__(self):
        """Initialize defaults and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if not 0 < self.mask_prob < 1:
            raise ValueError("mask_prob must be between 0 and 1")

        if not 0 <= self.mask_token_prob <= 1:
            raise ValueError("mask_token_prob must be between 0 and 1")

        if not 0 <= self.random_token_prob <= 1:
            raise ValueError("random_token_prob must be between 0 and 1")

        if self.mask_token_prob + self.random_token_prob > 1:
            raise ValueError(
                f"mask_token_prob ({self.mask_token_prob}) + "
                f"random_token_prob ({self.random_token_prob}) must be <= 1"
            )

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Set default embedding configs
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}
        
        if self.position_config is None:
            self.position_config = {"type": "learnable"}
        
        # Validate vocab_size to ensure mask token doesn't conflict
        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for PAD, MASK, and items), got {self.vocab_size}"
            )
