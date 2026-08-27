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
    dropout_rate: float = 0.2  # homogeneous neural-baseline dropout
    max_seq_length: int = 200
    vocab_size: int = None

    # Training (homogeneous neural-baseline recipe: lr=1e-3, wd=1e-6)
    lr: float = 1e-03
    weight_decay: float = 1e-06

    # Validation metrics (optional)
    compute_val_metrics: bool = False
    val_k_values: list = None
    metric_compute_interval: int = 10  # Compute NDCG and other metrics every N epochs

    # Masking strategy
    mask_prob: float = 0.4  # Cloze mask ratio (higher helps on short ML sequences)
    mask_token_prob: float = 0.8  # 80% replace with [MASK]
    random_token_prob: float = 0.1  # 10% replace with random token
    # Remaining 10% keep original token
    # Fraction of sequences where the LAST real item is force-masked, matching
    # the inference task (predict a trailing [MASK]). Bridges the train/eval gap.
    last_item_mask_prob: float = 0.5

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
    scale_embeddings: bool = True  # scale item embeddings by sqrt(d)
    use_bias: bool = False  # Bias on attention/FFN linears. Default False =
    # GPT/LLaMA bias-free convention (the field was previously dead + hardcoded
    # off). True matches bias-carrying BERT4Rec (original/RecBole).

    # --- RecBole-faithful toggles (default to current RecArena behavior) ---
    # RecBole BERT4Rec LayerNorms the input embeddings and has no final LN.
    input_layer_norm: bool = False  # LayerNorm on input embeddings (RecBole: True)
    final_layer_norm: bool = True  # LayerNorm after the blocks (RecBole: False)
    layer_norm_eps: float = 1e-5  # norm eps everywhere (RecBole BERT4Rec: 1e-12)

    # Output prediction head. Original BERT4Rec (and RecBole) apply a
    # Linear -> GELU -> LayerNorm transform to the encoder output BEFORE the tied
    # item scoring, plus a learned per-item output bias (a popularity prior).
    # RecArena historically scored the raw encoder output with no head/bias.
    # Enabling both is the standard architecture (original BERT4Rec / RecBole
    # ALWAYS have them), so the defaults are True to be RecBole-faithful since
    # the benchmark runs on defaults. Set False to score the raw encoder output.
    output_head: bool = True  # Linear->GELU->LayerNorm before tied scoring
    # Learned per-item additive logit bias (popularity prior). Now added to the
    # training logits too (via BERT4Rec.compute_loss -> CrossEntropyLoss
    # output_bias kwarg) so it is trained, matching inference. Note: with a
    # sampled loss the bias is still only applied at inference (the sampled path
    # forms its own logits); it is most meaningful with cross_entropy (default).
    output_bias: bool = True

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
