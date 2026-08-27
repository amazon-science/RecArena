"""GRU4Rec model configuration."""

from dataclasses import dataclass
from ..base import BaseModelConfig


@dataclass
class GRU4RecConfig(BaseModelConfig):
    """GRU4Rec configuration with defaults from original paper."""

    # Architecture
    embedding_dim: int = 64
    hidden_size: int = None  # defaults to embedding_dim (tied output requires match)
    num_layers: int = 1
    dropout_rate: float = 0.2
    max_seq_length: int = 200
    vocab_size: int = None

    # Training (homogeneous neural-baseline recipe: lr=1e-3, wd=1e-6)
    lr: float = 1e-03
    weight_decay: float = 1e-06

    # Validation metrics (optional)
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Loss function
    loss_type: str = "sampled_softmax"  # bpr, bce, gbce

    # GRU settings
    activation: str = "tanh"  # tanh, relu, gelu
    bidirectional: bool = False

    # Output projection. When True, a Linear(hidden_size -> embedding_dim) sits
    # between the GRU and the tied item scoring (the standard GRU4Rec design,
    # matching RecBole). This decouples hidden_size from embedding_dim so the GRU
    # can be wider than the embeddings. Default True matches RecBole, which
    # ALWAYS has the dense hidden->emb Linear (self.dense) and is the
    # weight-parity-proven configuration. Set False to score the raw GRU hidden
    # state directly (requires hidden_size == embedding_dim).
    #
    # NOTE ON GRU BIAS: RecArena's nn.GRU uses PyTorch's default bias=True,
    # whereas RecBole's GRU4Rec constructs nn.GRU(bias=False). This is left
    # as-is to avoid perturbing the weight-parity toggle; it is a minor
    # (bias-term) architectural difference, not a correctness bug.
    use_hidden_projection: bool = True

    # Initialization. `init_std` was previously declared but NEVER applied (the
    # item embedding fell back to PyTorch's default N(0,1), std=1.0 -- ~10x too
    # large, which hurt convergence). `apply_init` wires it: when True the item
    # (and position) embeddings are initialized ~N(0, init_std). Default True
    # fixes the long-standing uninitialized-embedding bug.
    init_std: float = 0.02
    apply_init: bool = True

    # Training objective:
    #   False (default) -> per-position causal-shift CE (predict item i+1 from
    #       every position i in ONE forward pass). More supervision per sequence;
    #       the modern SASRec-style objective RecArena shares across seq models.
    #   True -> last-position CE (gather the final hidden state, predict the
    #       single next item). Matches the canonical 2016 GRU4Rec recipe AND
    #       RecBole's GRU4Rec. Use to reproduce the paper/RecBole objective.
    last_position_loss: bool = False

    # Embedding configuration
    embedding_config: dict = None
    position_config: dict = None

    def __post_init__(self):
        """Initialize defaults and validate configuration."""
        # Initialize defaults
        if self.val_k_values is None:
            self.val_k_values = [10]

        # Without the output projection, tied scoring multiplies GRU hidden
        # states by the item embedding table, so hidden_size must equal
        # embedding_dim. With use_hidden_projection, a Linear bridges the two, so
        # hidden_size is free.
        if self.hidden_size is None:
            self.hidden_size = self.embedding_dim
        elif self.hidden_size != self.embedding_dim and not self.use_hidden_projection:
            raise ValueError(
                f"GRU4Rec hidden_size ({self.hidden_size}) must equal embedding_dim "
                f"({self.embedding_dim}) unless use_hidden_projection=True."
            )

        # Validate
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        # Set default embedding configs
        if self.embedding_config is None:
            self.embedding_config = {"type": "standard"}

        if self.position_config is None:
            self.position_config = {"type": "learnable"}

        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
