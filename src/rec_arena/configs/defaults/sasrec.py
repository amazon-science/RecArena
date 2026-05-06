"""SASRec model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class SASRecConfig(BaseModelConfig):
    """SASRec configuration with defaults from original paper."""

    # Architecture
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    feedforward_dim: int = (
        None  # Hidden dim in transformer FFN (default: 4 * embedding_dim)
    )
    dropout_rate: float = 0.2
    max_seq_length: int = 200
    vocab_size: int = None

    # Training
    lr: float = 1e-03
    weight_decay: float = 1e-06

    # Loss function
    loss_type: str = "sampled_softmax"  # bpr, bce, contrastive, focal, gbce
    temperature: float = 0.1  # for contrastive loss
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    # Validation metrics (optional)
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Architecture details
    activation: str = "gelu"  # gelu, relu, swish, tanh
    transformer_activation: str = "gelu"
    use_ligr: bool = False  # Use gated residuals + SwiGLU (LiGR style)
    use_rms_norm: bool = False  # Use RMSNorm instead of LayerNorm (LLaMA style)
    layer_norm_first: bool = True  # True=pre-norm (stable), False=post-norm (faster)
    scale_embeddings: bool = True  # Scale embeddings by sqrt(d) for better gradients
    init_std: float = 0.01
    use_bias: bool = True
    
    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None
    
    # Output layer configuration
    tie_embeddings: bool = True  # Tie input/output embeddings (default: tied)
    output_lora_rank: int = 0  # LoRA rank for output layer (0 = disabled)

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        # Default feedforward_dim to 4x embedding_dim (standard transformer practice)
        if self.feedforward_dim is None:
            self.feedforward_dim = 4 * self.embedding_dim

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        if self.feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive")

        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Set default embedding configs
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}
        
        if self.position_config is None:
            self.position_config = {"type": "learnable"}

        if self.vocab_size is not None and self.vocab_size <= 3:
            raise ValueError(
                f"vocab_size must be > 3 (reserved for PAD, UNK, MASK), got {self.vocab_size}"
            )

        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
