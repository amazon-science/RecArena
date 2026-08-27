import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

# Preferred CUDA SDPA backends, fastest first. FlashAttention-2 requires fp16/
# bf16 and head_dim <= 256; when it is ineligible (e.g. fp32, or an additive
# mask), SDPA transparently uses the memory-efficient kernel, and finally the
# math kernel as a correctness fallback. Listing all three lets one
# sdpa_kernel() context pick the best valid backend instead of crashing into a
# manual fallback.
_CUDA_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer for autoregressive models.
    Follows GPT/LLaMA style architecture with modern best practices.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        rope=None,
        qk_norm: bool = False,
        pos_bias=None,
        qk_norm_eps: float = 1e-6,
        bias: bool = False,
    ):
        """
        Initialize causal self-attention layer.

        :param dim: embedding dimension
        :param num_heads: number of attention heads
        :param dropout_rate: dropout probability
        :param rope: optional rotary embedding instance (RoPE or TO-RoPE). A
            TO-RoPE instance additionally consumes per-batch timestamps.
        :param qk_norm: if True, apply RMSNorm to per-head Q and K before
            attention (Gemma2/Qwen-style query-key normalization).
        :param pos_bias: optional additive attention-bias module (ALiBi or
            T5-RAB) returning ``[1, num_heads, S, S]`` given ``seq_len``. When
            set, attention runs on the (slower) additive-mask SDPA path.
        :param qk_norm_eps: eps for the QK RMSNorm.
        :param bias: if True, add bias terms to the QKV and output projections.
            Default False (the GPT/LLaMA-style bias-free convention this codebase
            uses). Set True to match implementations that keep projection biases
            (e.g. the original BERT/RecBole SASRec), enabling exact weight-bridge
            equivalence checks.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.rope = rope
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.pos_bias = pos_bias

        # QKV projection (combined for efficiency)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias)
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.dropout_rate = dropout_rate

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for causal self-attention.

        :param x: input tensor of shape [batch_size, seq_len, dim]
        :param attn_mask: optional attention mask of shape [batch_size, seq_len]
            (key padding mask, True = keep) or [batch_size, seq_len, seq_len]
            (full additive/boolean mask)
        :param is_causal: whether to apply causal masking (default: True)
        :return: output tensor of shape [batch_size, seq_len, dim]

        FlashAttention note: PyTorch's fused SDPA only dispatches to the
        FlashAttention-2 backend on CUDA, and Flash does not support arbitrary
        additive masks. We therefore keep the fast path (``is_causal=True`` with
        ``attn_mask=None``) whenever possible; passing an explicit mask forces
        the slower memory-efficient/math kernel. On CPU/MPS, SDPA always uses a
        non-Flash backend.
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # [3, batch_size, num_heads, seq_len, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # QK-norm: RMSNorm over the per-head feature dim of Q and K before
        # rotary/attention (Gemma2/Qwen). Stabilizes attention logits and is a
        # cheap add-on tested here on top of the RoPE+LiGR anchor.
        if self.qk_norm:
            Q = F.rms_norm(Q, normalized_shape=[self.head_dim], eps=self.qk_norm_eps)
            K = F.rms_norm(K, normalized_shape=[self.head_dim], eps=self.qk_norm_eps)

        # Apply rotary embedding if available. TO-RoPE additionally consumes
        # per-batch timestamps; vanilla RoPE ignores them.
        if self.rope is not None:
            from ..layer_utils.embeddings import TimeOrderRotaryEmbedding

            if isinstance(self.rope, TimeOrderRotaryEmbedding):
                Q, K = self.rope(Q, K, seq_len, timestamps=timestamps)
            else:
                Q, K = self.rope(Q, K, seq_len)

        # Additive attention-bias position encodings (ALiBi / T5-RAB) produce an
        # additive [1, H, S, S] bias. SDPA accepts a float additive mask, so we
        # fold the causal structure into the same tensor (masked entries -> -inf)
        # and pass it as the SDPA attn_mask. This forces the memory-efficient/
        # math backend (Flash rejects additive masks), which is expected for
        # these variants.
        if self.pos_bias is not None:
            bias = self.pos_bias(seq_len).to(Q.dtype)  # [1, H, S, S]
            add_mask = bias.expand(batch_size, self.num_heads, seq_len, seq_len).clone()
            if is_causal:
                causal_block = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
                add_mask = add_mask.masked_fill(
                    causal_block.view(1, 1, seq_len, seq_len), float("-inf")
                )
            if attn_mask is not None and attn_mask.dim() == 2:
                key_pad = ~attn_mask.bool().view(batch_size, 1, 1, seq_len)
                add_mask = add_mask.masked_fill(key_pad, float("-inf"))
            dropout_p = self.dropout_rate if self.training else 0.0
            output = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=add_mask, dropout_p=dropout_p, is_causal=False
            )
            output = (
                output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            )
            return self.out_proj(output)

        # Build the SDPA mask. We prefer the fused causal flag (Flash-friendly)
        # and only materialize a boolean mask when a custom mask is supplied.
        sdpa_mask = None
        use_causal_flag = is_causal
        if attn_mask is not None:
            use_causal_flag = False  # custom mask supersedes the causal flag
            if attn_mask.dim() == 2:
                # Key padding mask [B, S] (True = keep) -> boolean [B, 1, 1, S].
                # Combine with causal structure when causal is requested so we
                # don't lose autoregressive masking.
                key_keep = attn_mask.bool().view(batch_size, 1, 1, seq_len)
                if is_causal:
                    causal_keep = torch.tril(
                        torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
                    ).view(1, 1, seq_len, seq_len)
                    sdpa_mask = key_keep & causal_keep
                else:
                    sdpa_mask = key_keep.expand(batch_size, 1, seq_len, seq_len)
            elif attn_mask.dim() == 3:
                # Full mask [B, S, S] -> [B, 1, S, S]. Boolean = keep; float = additive.
                sdpa_mask = attn_mask.unsqueeze(1)

        dropout_p = self.dropout_rate if self.training else 0.0
        if Q.device.type == "cuda":
            # Offer Flash -> mem-efficient -> math; SDPA picks the fastest valid
            # backend for this dtype/shape/mask (Flash needs fp16/bf16).
            with sdpa_kernel(_CUDA_SDPA_BACKENDS):
                output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=use_causal_flag and sdpa_mask is None,
                )
        else:
            # CPU/MPS: SDPA uses a non-Flash backend.
            output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=use_causal_flag and sdpa_mask is None,
            )

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer for cross-attention or non-causal self-attention.
    Follows modern transformer architecture standards.
    """

    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Initialize multi-head attention layer.

        :param dim: embedding dimension
        :param num_heads: number of attention heads
        :param dropout_rate: dropout probability
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        # Separate Q, K, V projections for cross-attention
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.val_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout_rate = dropout_rate

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        :param query: query tensor of shape [batch_size, seq_len_q, dim]
        :param key: key tensor of shape [batch_size, seq_len_k, dim]
        :param value: value tensor of shape [batch_size, seq_len_k, dim]
        :param attn_mask: optional attention mask
        :param is_causal: whether to apply causal masking (default: False)
        :return: output tensor of shape [batch_size, seq_len_q, dim]
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]

        # Project Q, K, V
        Q = (
            self.query_proj(query)
            .view(batch_size, seq_len_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key_proj(key)
            .view(batch_size, seq_len_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.val_proj(value)
            .view(batch_size, seq_len_k, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply scaled dot-product attention
        if Q.device.type == "cuda":
            with sdpa_kernel(_CUDA_SDPA_BACKENDS):
                output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_rate if self.training else 0.0,
                    is_causal=is_causal,
                )
        else:
            output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal,
            )

        # Reshape and apply output projection
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.dim)
        )
        output = self.out_proj(output)

        return output
