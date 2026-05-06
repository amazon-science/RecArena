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

from .regularization import (
    ContrastiveLoss,
    FocalLoss,
    LabelSmoothingLoss,
    MultiTaskLoss,
)
from .factory import get_loss_function

__all__ = [
    # Sequential
    "CrossEntropyLoss",
    "BCENegativeSamplingLoss",
    "SampledSoftmaxLoss",
    "SequentialBPRLoss",
    "GBCE",
    # Implicit
    "BCELoss",
    "BPRLoss",
    # Other
    "ContrastiveLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "MultiTaskLoss",
    "get_loss_function",
]
