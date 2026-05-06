import torch
from torch import nn
from .mha import CausalSelfAttention, MultiHeadAttention
from ..layer_utils.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout_rate: float = 0.5,
        causality: bool = True,
        activation: nn.Module = None,
        use_swiglu: bool = False,
        use_rms_norm: bool = False,
        use_gated_residual: bool = False,
        norm_first: bool = True,
        rope=None,
    ):
        """
        Modern Pre-LN transformer block following GPT/LLaMA architecture.

        :param dim: embedding dim
        :param num_heads: number of heads
        :param hidden_dim: dimensionality of the feed-forward layers
        :param dropout_rate: dropout rate
        :param causality: bool, if True the causal mask is applied (i.e. for next-item prediction)
        :param activation: activation module (e.g., nn.GELU(), nn.ReLU(), nn.Tanh(), nn.SiLU())
        :param use_swiglu: if True, use SwiGLU activation (modern LLM standard)
        :param use_rms_norm: if True, use RMSNorm instead of LayerNorm (LLaMA style)
        :param use_gated_residual: if True, use gated residuals (LiGR style)
        """
        super().__init__()

        # Choose normalization type
        if use_rms_norm:
            self.attn_norm = RMSNorm(dim)
            self.ffn_norm = RMSNorm(dim)
        else:
            self.attn_norm = nn.LayerNorm(dim)
            self.ffn_norm = nn.LayerNorm(dim)

        self.attention = CausalSelfAttention(dim, num_heads, dropout_rate, rope=rope)

        # Choose FFN architecture
        self.use_swiglu = use_swiglu
        if use_swiglu:
            # SwiGLU: gated feed-forward network (modern LLM standard)
            self.ffn = SwiGLU(dim, hidden_dim, dropout_rate, bias=False)
        else:
            self.dense1 = nn.Linear(dim, hidden_dim, bias=False)
            self.dense2 = nn.Linear(hidden_dim, dim, bias=False)
            self.activation = activation if activation is not None else nn.GELU()

        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality
        self.use_gated_residual = use_gated_residual
        self.norm_first = norm_first

        # Gating layers for residual connections (LiGR style)
        if use_gated_residual:
            self.gate_attn = nn.Linear(dim, dim)
            self.gate_ffn = nn.Linear(dim, dim)

    def forward(self, seq: torch.Tensor, attn_mask: torch.Tensor | None = None):
        # Attention block
        residual = seq
        x = self.attn_norm(seq) if self.norm_first else seq
        x = self.attention(x, attn_mask=attn_mask, is_causal=self.causality)
        x = self.dropout(x)
        x = residual + (torch.sigmoid(self.gate_attn(residual)) * x if self.use_gated_residual else x)
        x = x if self.norm_first else self.attn_norm(x)

        # Feed-forward block
        residual = x
        x = self.ffn_norm(x) if self.norm_first else x
        
        if self.use_swiglu:
            x = self.ffn(x)
        else:
            x = self.dense1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.dense2(x)
            x = self.dropout(x)

        x = residual + (torch.sigmoid(self.gate_ffn(residual)) * x if self.use_gated_residual else x)
        x = x if self.norm_first else self.ffn_norm(x)

        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed
