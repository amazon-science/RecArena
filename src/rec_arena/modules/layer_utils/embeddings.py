"""Hierarchical embedding with per-item LoRA adapters."""

import torch
from torch import nn


class HierarchicalLoRAEmbedding(nn.Module):
    """
    Embedding layer that uses hierarchical embeddings as base with per-item LoRA adapters.

    Architecture:
    - Base: Parent embeddings (num_parents, embedding_dim)
    - Adapter: Per-item LoRA (A: num_items x rank, B: rank x embedding_dim)
    - Output: parent_emb[parent_of_item] + A[item] @ B

    This dramatically reduces parameters when num_items >> num_parents.
    Example: 5M tracks, 2M artists, dim=128, rank=16
    - Regular embedding: 5M × 128 = 640M parameters
    - Hierarchical+LoRA: 2M × 128 + 5M × 16 + 16 × 128 = 256M + 80M + 2K ≈ 336M parameters
    """

    def __init__(
        self,
        num_items: int,
        num_parents: int,
        embedding_dim: int,
        item_to_parent_mapping: torch.Tensor,
        lora_rank: int = 16,
        scale_grad_by_freq: bool = False,
    ):
        """
        Args:
            num_items: Total number of items
            num_parents: Total number of parent entities
            embedding_dim: Dimension of embeddings
            item_to_parent_mapping: Tensor of shape [num_items] mapping item_id -> parent_id
            lora_rank: Rank of LoRA adapter matrices
            scale_grad_by_freq: Whether to scale gradients by frequency
        """
        super().__init__()

        self.num_items = num_items
        self.num_parents = num_parents
        self.embedding_dim = embedding_dim
        self.lora_rank = lora_rank

        # Parent embeddings (trainable)
        self.parent_embedding = nn.Embedding(
            num_parents, embedding_dim, scale_grad_by_freq=scale_grad_by_freq
        )

        # Item-to-parent mapping (frozen)
        self.register_buffer("item_to_parent", item_to_parent_mapping)

        # Per-item LoRA adapters - use proper initialization
        # LoRA_A: Kaiming uniform for good gradient flow
        # LoRA_B: zeros (standard LoRA initialization)
        self.lora_A = nn.Parameter(torch.empty(num_items, lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.lora_B = nn.Parameter(torch.zeros(lora_rank, embedding_dim))

    @property
    def weight(self):
        """Reconstruct full embedding matrix for all items."""
        # Get parent embeddings for all items
        parent_embs = self.parent_embedding(
            self.item_to_parent
        )  # [num_items, embedding_dim]
        # Add LoRA adaptation
        lora_adaptation = self.lora_A @ self.lora_B  # [num_items, embedding_dim]
        return parent_embs + lora_adaptation

    @property
    def num_embeddings(self) -> int:
        """Alias for num_items to satisfy the ItemEmbedding protocol."""
        return self.num_items

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical+LoRA embedding.

        Args:
            item_ids: Tensor of item indices, any shape

        Returns:
            Embeddings of shape [..., embedding_dim]
        """
        # Get parent IDs for these items
        parent_ids = self.item_to_parent[item_ids.long()]  # Same shape as item_ids

        # Get base parent embeddings
        parent_embs = self.parent_embedding(parent_ids)  # [..., embedding_dim]

        # Get LoRA adaptation for these specific items
        lora_A_items = nn.functional.embedding(
            item_ids.long(), self.lora_A
        )  # [..., rank]
        lora_adaptation = lora_A_items @ self.lora_B  # [..., embedding_dim]

        return parent_embs + lora_adaptation


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer paper
    Paper: https://arxiv.org/abs/2104.09864
    Used in GPT-NeoX, LLaMA, and most modern LLMs
    Better extrapolation and relative position encoding
    """

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len):
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: queries of shape (batch, n_heads, seq_len, head_dim)
            k: keys of shape (batch, n_heads, seq_len, head_dim)
            seq_len: sequence length

        Layout note: the cos/sin caches are built as [1, S, 1, Hd] then
        transposed to [1, 1, S, Hd] so they broadcast over (batch, n_heads) and
        align on the seq axis of the [B, H, S, Hd] inputs used here. This is
        verified by tests/test_transformer_core.py (relative-position
        invariance of <RoPE(q,m), RoPE(k,n)>). Changing the q/k layout requires
        updating this broadcast.
        """
        # Get cos and sin for the sequence length
        # Shape: (1, seq_len, 1, head_dim)
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]

        # Reshape to match q and k: (1, 1, seq_len, head_dim) -> broadcast to (batch, n_heads, seq_len, head_dim)
        cos = cos.transpose(1, 2)  # (1, 1, seq_len, head_dim)
        sin = sin.transpose(1, 2)  # (1, 1, seq_len, head_dim)

        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


class TimeOrderRotaryEmbedding(nn.Module):
    """Time-and-Order RoPE (TO-RoPE) for generative recommendation.

    Standard RoPE encodes only discrete sequence *order* (position index m).
    TO-RoPE (Wei et al. 2025, arXiv:2510.20455) additionally encodes wall-clock
    *time* by splitting the head dimension into two rotary sub-blocks:

      - the ORDER block rotates by the position index m (as in vanilla RoPE);
      - the TIME block rotates by a normalized timestamp phase.

    Both blocks retain RoPE's relative-encoding property (the q-k dot product
    depends only on Δorder and Δtime), so the model sees both "how many items
    ago" and "how long ago" an interaction occurred. When timestamps are
    unavailable the time block falls back to the position index, recovering
    vanilla RoPE.

    ``time_ratio`` is the fraction of rotary dimensions assigned to the time
    block (default 0.5, an even order/time split).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        time_ratio: float = 0.5,
        time_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.time_scale = time_scale

        # Split the rotary pairs into a time block and an order block. dim is the
        # head dim; RoPE operates on dim/2 frequency pairs. We assign the first
        # `n_time` pairs to time and the rest to order.
        n_pairs = dim // 2
        self.n_time_pairs = max(1, min(n_pairs - 1, int(round(n_pairs * time_ratio))))
        self.n_order_pairs = n_pairs - self.n_time_pairs

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq[:n_time_pairs] -> time block, inv_freq[n_time_pairs:] -> order.
        self.register_buffer("inv_freq", inv_freq)

        # Precompute order-block cos/sin over positions (fixed integer index).
        t = torch.arange(max_seq_len, dtype=torch.float32)
        order_freqs = torch.outer(t, inv_freq[self.n_time_pairs :])  # [S, n_order]
        self.register_buffer("_order_cos", order_freqs.cos(), persistent=False)
        self.register_buffer("_order_sin", order_freqs.sin(), persistent=False)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _build_cos_sin(self, seq_len, timestamps, device, dtype):
        """Assemble per-position [.., S, dim] cos/sin from time + order blocks.

        timestamps: [B, S] real interaction times (float) or None. When present
        the time block uses per-batch normalized timestamps (shift so the first
        valid step is 0, scale by ``time_scale``); otherwise it falls back to the
        position index (== vanilla RoPE).
        """
        order_cos = self._order_cos[:seq_len].to(device=device, dtype=torch.float32)
        order_sin = self._order_sin[:seq_len].to(device=device, dtype=torch.float32)

        time_inv = self.inv_freq[: self.n_time_pairs].to(device)

        if timestamps is not None:
            ts = timestamps.to(device=device, dtype=torch.float32)  # [B, S]
            ts = ts - ts[:, :1]  # relative to the first position, per sequence
            ts = ts * self.time_scale
            # [B, S, n_time]
            time_phase = ts.unsqueeze(-1) * time_inv.view(1, 1, -1)
            time_cos, time_sin = time_phase.cos(), time_phase.sin()
            B = ts.size(0)
            order_cos_b = order_cos.unsqueeze(0).expand(B, -1, -1)
            order_sin_b = order_sin.unsqueeze(0).expand(B, -1, -1)
            half_cos = torch.cat([time_cos, order_cos_b], dim=-1)  # [B, S, n_pairs]
            half_sin = torch.cat([time_sin, order_sin_b], dim=-1)
            cos = torch.cat([half_cos, half_cos], dim=-1)  # [B, S, dim]
            sin = torch.cat([half_sin, half_sin], dim=-1)
            # [B, 1, S, dim] to broadcast over heads.
            return cos.unsqueeze(1).to(dtype), sin.unsqueeze(1).to(dtype)

        # No timestamps: time block uses the position index (recovers RoPE).
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        time_phase = torch.outer(pos, time_inv)  # [S, n_time]
        half_cos = torch.cat([time_phase.cos(), order_cos], dim=-1)  # [S, n_pairs]
        half_sin = torch.cat([time_phase.sin(), order_sin], dim=-1)
        cos = torch.cat([half_cos, half_cos], dim=-1)  # [S, dim]
        sin = torch.cat([half_sin, half_sin], dim=-1)
        return (
            cos[None, None, :, :].to(dtype),
            sin[None, None, :, :].to(dtype),
        )

    def forward(self, q, k, seq_len, timestamps=None):
        """Apply time+order rotary embeddings to q, k of shape [B, H, S, Hd]."""
        cos, sin = self._build_cos_sin(seq_len, timestamps, q.device, q.dtype)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
