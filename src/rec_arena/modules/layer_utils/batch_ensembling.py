import math
from typing import Literal

import torch
from torch import nn


class LinearBatchEnsembleLayer(nn.Module):
    """A configurable BatchEnsemble layer that supports optional input scaling, output scaling,
    and output bias terms as per the 'BatchEnsemble' paper.
    It provides initialization options for scaling terms to diversify ensemble members.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        ensemble_scaling_in: bool = True,
        ensemble_scaling_out: bool = True,
        ensemble_bias: bool = False,
        scaling_init: Literal["ones", "random-signs", "normal"] = "ones",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Base weight matrix W, shared across ensemble members
        self.W = nn.Parameter(torch.empty(out_features, in_features))

        # Optional scaling factors and shifts for each ensemble member
        self.r = (
            nn.Parameter(torch.empty(ensemble_size, in_features))
            if ensemble_scaling_in
            else None
        )
        self.s = (
            nn.Parameter(torch.empty(ensemble_size, out_features))
            if ensemble_scaling_out
            else None
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features))
            if not ensemble_bias and out_features > 0
            else (
                nn.Parameter(torch.empty(ensemble_size, out_features))
                if ensemble_bias
                else None
            )
        )

        # Initialize parameters
        self.reset_parameters(scaling_init)

    def reset_parameters(self, scaling_init: Literal["ones", "random-signs", "normal"]):
        # Initialize W using Xavier uniform for stability
        nn.init.xavier_uniform_(self.W)

        # Initialize scaling factors r and s based on selected initialization
        def random_signs_init(x):
            """Initialize with random signs, ensuring no zeros."""
            signs = torch.sign(torch.randn_like(x))
            # Replace any zeros with 1 (extremely rare but possible)
            signs[signs == 0] = 1.0
            x.data.copy_(signs)
            return x

        scaling_init_fn = {
            "ones": nn.init.ones_,
            "random-signs": random_signs_init,
            "normal": lambda x: nn.init.normal_(x, mean=1.0, std=0.1),  # Mean 1 for stability
        }

        if self.r is not None:
            scaling_init_fn[scaling_init](self.r)
        if self.s is not None:
            scaling_init_fn[scaling_init](self.s)

        # Initialize bias
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # noqa: PLR2004
            # Input: (B, S, D) -> Output: (B, S, E, D)
            B, S, D = x.shape  # noqa: N806
            # Expand to (B, S, E, D)
            x = x.unsqueeze(2).expand(B, S, self.ensemble_size, D)
        elif x.dim() == 2:  # noqa: PLR2004
            # Input: (B, D) -> Output: (B, E, D)
            x = x.unsqueeze(1).expand(-1, self.ensemble_size, -1)
        elif x.dim() == 4 and x.size(2) != self.ensemble_size:  # noqa: PLR2004
            shape_error = f"Input shape {x.shape} is invalid. Expected ensemble size {self.ensemble_size}"
            raise ValueError(shape_error)

        # Apply input scaling if enabled
        if self.r is not None:
            x = x * self.r

        # Linear transformation with W - use einsum for efficient parallel computation
        output = torch.einsum("...ed,od->...eo", x, self.W)

        # Apply output scaling if enabled
        if self.s is not None:
            output = output * self.s

        # Add bias if enabled
        if self.bias is not None:
            output = output + self.bias

        return output
