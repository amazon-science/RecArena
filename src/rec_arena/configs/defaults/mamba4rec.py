"""Mamba4Rec model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class Mamba4RecConfig(BaseModelConfig):
    """Mamba4Rec configuration with defaults."""

    # Architecture
    embedding_dim: int = 64
    d_model: int = 64  # Mamba hidden dimension
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4    # Local convolution width
    expand_factor: int = 2  # Expansion factor
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

    # Loss function
    loss_type: str = "sampled_softmax"

    # Mamba settings
    mamba_version: str = "mamba2"  # mamba1 or mamba2
    norm: str = "LayerNorm"  # Normalization layer
    bidirectional: bool = False
    dt_min: float = 1e-04
    dt_max: float = 0.1
    dt_init_floor: float = 1e-04
    conv_bias: bool = True
    bias: bool = False

    # Initialization
    init_std: float = 0.02

    
    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None
    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.val_k_values is None:
            self.val_k_values = [10]

        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Set default embedding configs
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}
        
        if self.position_config is None:
            self.position_config = {"type": "learnable"}

        if self.d_model < 1:
            raise ValueError("d_model must be at least 1")

        if self.mamba_version not in ["mamba1", "mamba2"]:
            raise ValueError("mamba_version must be 'mamba1' or 'mamba2'")