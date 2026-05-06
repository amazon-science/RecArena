import torch
from torch import nn


class FeedForward(nn.Module):
    """
    Standard feed-forward network with configurable activation.

    Parameters
    ----------
    n_factors : int
        Input/output dimension.
    n_factors_ff : int
        Hidden dimension.
    dropout_rate : float
        Dropout probability.
    activation : nn.Module
        Activation function (e.g., nn.GELU(), nn.ReLU()).
    bias : bool
        Whether to use bias in linear layers.
    """

    def __init__(
        self,
        n_factors: int,
        n_factors_ff: int,
        dropout_rate: float,
        activation: nn.Module,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(n_factors, n_factors_ff, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(n_factors_ff, n_factors, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class SwiGLU(nn.Module):
    """
    SwiGLU gated feed-forward network.
    From FuXi and LLaMA: https://arxiv.org/pdf/2502.03036
    From LiGR: https://arxiv.org/pdf/2502.03417

    Parameters
    ----------
    n_factors : int
        Input/output dimension.
    n_factors_ff : int
        Hidden dimension.
    dropout_rate : float
        Dropout probability.
    bias : bool
        Whether to use bias in linear layers.
    """

    def __init__(
        self, n_factors: int, n_factors_ff: int, dropout_rate: float, bias: bool = True
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(n_factors, n_factors_ff, bias=bias)
        self.linear2 = nn.Linear(n_factors_ff, n_factors, bias=bias)
        self.gate = nn.Linear(n_factors, n_factors_ff, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(
            self.dropout(nn.functional.silu(self.linear1(x)) * self.gate(x))
        )
