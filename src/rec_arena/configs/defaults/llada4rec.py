"""LLaDA4Rec model configuration."""

from dataclasses import dataclass
from .bert4rec import BERT4RecConfig


@dataclass
class LLaDA4RecConfig(BERT4RecConfig):
    """LLaDA4Rec configuration - diffusion-based sequential recommendation.
    
    Extends BERT4Rec with variable masking ratio and iterative generation.
    """

    # Diffusion-specific parameters
    eps: float = 0.01  # Minimum masking ratio (avoid 0 for numerical stability)
    diffusion_steps: int = 50  # Number of iterative unmasking steps during inference
    remasking_strategy: str = "low_confidence"  # "low_confidence" or "random"
    temperature: float = 0.0  # Sampling temperature (0 = greedy)
    
    # Override BERT4Rec defaults for LLaDA
    mask_token_prob: float = 1.0  # Always use mask token (no random replacement)
    random_token_prob: float = 0.0  # No random token replacement
    
    def __post_init__(self):
        """Validate LLaDA-specific configuration."""
        super().__post_init__()
        
        if not 0 < self.eps < 1:
            raise ValueError(f"eps must be in (0, 1), got {self.eps}")
        
        if self.diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be positive, got {self.diffusion_steps}")
        
        if self.remasking_strategy not in ["low_confidence", "random"]:
            raise ValueError(
                f"remasking_strategy must be 'low_confidence' or 'random', got {self.remasking_strategy}"
            )
        
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
