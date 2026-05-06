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
        Apply rotary embeddings to queries and keys
        Args:
            q: queries of shape (batch, n_heads, seq_len, head_dim)
            k: keys of shape (batch, n_heads, seq_len, head_dim)
            seq_len: sequence length
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
