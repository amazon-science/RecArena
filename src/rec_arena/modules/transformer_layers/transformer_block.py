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
        qk_norm: bool = False,
        pos_bias=None,
        peri_norm: bool = False,
        bias: bool = False,
        norm_eps: float = 1e-5,
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
        :param qk_norm: if True, RMSNorm Q/K before attention (passed to attn).
        :param pos_bias: optional additive position-bias module (ALiBi/T5-RAB).
        :param peri_norm: if True, apply a "sandwich"/peri-norm: normalize the
            sublayer input (pre-norm) AND the sublayer output before the residual
            add. Adds a second norm per sublayer; a third norm-position variant
            beyond pre-norm (norm_first=True) and post-norm (norm_first=False).
        :param bias: if True, add bias terms to the attention projections and the
            FFN linears. Default False (GPT/LLaMA bias-free convention). Set True
            to match bias-carrying implementations (original BERT/RecBole SASRec)
            for exact weight-bridge equivalence.
        """
        super().__init__()

        self.peri_norm = peri_norm

        # Choose normalization type
        def _make_norm():
            return RMSNorm(dim, eps=norm_eps) if use_rms_norm else nn.LayerNorm(dim, eps=norm_eps)

        self.attn_norm = _make_norm()
        self.ffn_norm = _make_norm()
        # Peri-norm adds a post-sublayer normalization before each residual add.
        if peri_norm:
            self.attn_post_norm = _make_norm()
            self.ffn_post_norm = _make_norm()

        self.attention = CausalSelfAttention(
            dim,
            num_heads,
            dropout_rate,
            rope=rope,
            qk_norm=qk_norm,
            pos_bias=pos_bias,
            bias=bias,
        )

        # Choose FFN architecture
        self.use_swiglu = use_swiglu
        if use_swiglu:
            # SwiGLU: gated feed-forward network (modern LLM standard)
            self.ffn = SwiGLU(dim, hidden_dim, dropout_rate, bias=bias)
        else:
            self.dense1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.dense2 = nn.Linear(hidden_dim, dim, bias=bias)
            self.activation = activation if activation is not None else nn.GELU()

        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality
        self.use_gated_residual = use_gated_residual
        self.norm_first = norm_first

        # Gating layers for residual connections (LiGR style).
        # LiGR (arXiv:2502.03417) gates each sublayer output by a sigmoid of a
        # linear projection of the block input:  h_{j+1} = h_j + F(h_j) * sigma(h_j W).
        # Here F(h_j) is the (pre-normed) sublayer output and the gate sees the
        # residual stream h_j.
        if use_gated_residual:
            self.gate_attn = nn.Linear(dim, dim)
            self.gate_ffn = nn.Linear(dim, dim)

    def forward(
        self,
        seq: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ):
        # Attention block
        residual = seq
        x = self.attn_norm(seq) if (self.norm_first or self.peri_norm) else seq
        x = self.attention(
            x, attn_mask=attn_mask, is_causal=self.causality, timestamps=timestamps
        )
        x = self.dropout(x)
        # Peri-norm: normalize the sublayer output before the residual add.
        if self.peri_norm:
            x = self.attn_post_norm(x)
        x = residual + (
            torch.sigmoid(self.gate_attn(residual)) * x
            if self.use_gated_residual
            else x
        )
        # Post-norm (only when neither pre- nor peri-norm handled it).
        x = x if (self.norm_first or self.peri_norm) else self.attn_norm(x)

        # Feed-forward block
        residual = x
        x = self.ffn_norm(x) if (self.norm_first or self.peri_norm) else x

        if self.use_swiglu:
            x = self.ffn(x)
        else:
            x = self.dense1(x)
            x = self.activation(x)
            x = self.dense2(x)
            # Single FFN dropout AFTER dense2 only, matching RecBole's
            # FeedForward (layers.py:526-533). A prior extra dropout on the
            # post-activation inner state double-regularized the FFN (at
            # dropout=0.5 that is heavy extra stochasticity on every layer's
            # inner activation), depressing the converged optimum vs RecBole.
            # Invisible to eval-mode weight-parity tests; active in training.
            x = self.dropout(x)

        if self.peri_norm:
            x = self.ffn_post_norm(x)
        x = residual + (
            torch.sigmoid(self.gate_ffn(residual)) * x if self.use_gated_residual else x
        )
        x = x if (self.norm_first or self.peri_norm) else self.ffn_norm(x)

        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style).

    Thin wrapper over ``torch.nn.functional.rms_norm`` so the whole codebase
    shares one RMSNorm definition and eps convention (eps added to the mean of
    squares, inside the rsqrt), matching the FuXi/HSTU models which call
    ``F.rms_norm`` directly.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x, normalized_shape=[self.dim], weight=self.weight, eps=self.eps
        )
