# Sequential losses
from .sequential import (
    CrossEntropyLoss,
    BCENegativeSamplingLoss,
    SampledSoftmaxLoss,
    BPRLoss as SequentialBPRLoss,
    GBCE,
)

# Implicit losses
from .implicit import (
    BCELoss,
    BPRLoss,
)
from .regularization import ContrastiveLoss, FocalLoss, LabelSmoothingLoss


def get_loss_function(loss_type: str, model_type: str = "sequential", **kwargs):
    """Factory function to create loss functions from string names."""
    # Validate inputs
    if not isinstance(loss_type, str) or not loss_type:
        raise ValueError("loss_type must be a non-empty string")
    if model_type not in ["sequential", "implicit"]:
        raise ValueError(
            f"model_type must be 'sequential' or 'implicit', got '{model_type}'"
        )

    # Sequential model losses
    sequential_losses = {
        "cross_entropy": CrossEntropyLoss,
        "bce": BCENegativeSamplingLoss,
        "sampled_softmax": SampledSoftmaxLoss,
        "bpr": SequentialBPRLoss,
        "gbce": GBCE,
    }

    # Implicit model losses
    implicit_losses = {
        "bce": BCELoss,
        "bpr": BPRLoss,
    }

    # Other losses
    other_losses = {
        "contrastive": ContrastiveLoss,
        "focal": FocalLoss,
        "label_smoothing": LabelSmoothingLoss,
    }

    # Validate loss-model compatibility
    incompatible_losses = ["cross_entropy", "sampled_softmax", "gbce"]
    if model_type == "implicit" and loss_type in incompatible_losses:
        raise ValueError(
            f"Loss '{loss_type}' not compatible with implicit models. Use 'bce' or 'bpr'"
        )

    # Select appropriate loss map based on model type
    if model_type == "implicit":
        loss_map = implicit_losses
    else:
        loss_map = {**sequential_losses, **other_losses}

    if loss_type not in loss_map:
        available = list(loss_map.keys())
        raise ValueError(
            f"Unknown loss type '{loss_type}' for model type '{model_type}'. Available: {available}"
        )

    return loss_map[loss_type](**kwargs)
