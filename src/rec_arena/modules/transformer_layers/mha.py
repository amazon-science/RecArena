import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer for autoregressive models.
    Follows GPT/LLaMA style architecture with modern best practices.
    """

    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1, rope=None):
        """
        Initialize causal self-attention layer.

        :param dim: embedding dimension
        :param num_heads: number of attention heads
        :param dropout_rate: dropout probability
        :param rope: optional RoPE instance
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.rope = rope

        # QKV projection (combined for efficiency)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout_rate = dropout_rate

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for causal self-attention.

        :param x: input tensor of shape [batch_size, seq_len, dim]
        :param attn_mask: optional attention mask of shape [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        :param is_causal: whether to apply causal masking (default: True for autoregressive)
        :return: output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # [3, batch_size, num_heads, seq_len, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if available
        if self.rope is not None:
            Q, K = self.rope(Q, K, seq_len)

        # Prepare attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Padding mask: [batch_size, seq_len]
                # Convert to attention mask: [batch_size, 1, 1, seq_len]
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(batch_size, 1, seq_len, seq_len)
                # Convert boolean mask to float mask
                attn_mask = torch.where(
                    attn_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
                )
            elif attn_mask.dim() == 3:
                # Full attention mask: [batch_size, seq_len, seq_len]
                attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # Apply scaled dot-product attention with Flash Attention when available
        if Q.device.type == "cuda":
            try:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = F.scaled_dot_product_attention(
                        Q,
                        K,
                        V,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        is_causal=is_causal
                        and attn_mask is None,  # Only use is_causal if no custom mask
                    )
            except RuntimeError:
                # Fallback if Flash Attention not available
                output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_rate if self.training else 0.0,
                    is_causal=is_causal and attn_mask is None,
                )
        else:
            # CPU/MPS fallback
            output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=is_causal and attn_mask is None,
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
            try:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = F.scaled_dot_product_attention(
                        Q,
                        K,
                        V,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        is_causal=is_causal,
                    )
            except RuntimeError:
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
