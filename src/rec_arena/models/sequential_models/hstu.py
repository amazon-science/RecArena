import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from ..sequential import DeepSequentialModel
from ...configs.defaults.hstu import HSTUConfig


class RelativePositionalBias(nn.Module):
    """Learnable relative positional bias."""

    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self._w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02)
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """Compute relative position bias. Returns [1, N, N]."""
        n = self.max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class PointwiseAggregatedAttention(nn.Module):
    """Pointwise aggregated attention with SiLU activation (HSTU-style)."""

    def __init__(self, embedding_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.rel_pos_bias = RelativePositionalBias(max_seq_len)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
            ),
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Full attention. Args: q, k, v: [B, N, D]. Returns: [B, N, D]"""
        B, N, D = q.shape

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # No sqrt(d) scaling - reference doesn't use it
        scores = torch.matmul(q, k.transpose(-2, -1))
        rel_bias = self.rel_pos_bias(N)
        scores = scores + rel_bias.unsqueeze(1)

        # Apply SiLU and normalize FIRST, then mask
        attn = F.silu(scores) / seq_len
        
        # Apply causal mask by multiplication (not masked_fill)
        causal_mask = self.causal_mask[:N, :N].unsqueeze(0).unsqueeze(0)
        attn = attn * (~causal_mask).float()

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        return output

    def cached_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cached_k: Optional[torch.Tensor],
        cached_v: Optional[torch.Tensor],
        pos: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental forward with KV cache. Args: q, k, v: [B, 1, D]. Returns: output [B, 1, D], new_k, new_v"""
        B, _, D = q.shape

        # Concatenate with cache
        if cached_k is None:
            full_k = k
            full_v = v
        else:
            full_k = torch.cat([cached_k, k], dim=1)
            full_v = torch.cat([cached_v, v], dim=1)

        # Reshape for attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        full_k = full_k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        full_v = full_v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with full history (no sqrt(d) scaling)
        scores = torch.matmul(q, full_k.transpose(-2, -1))
        rel_bias = self.rel_pos_bias(pos + 1)[:, pos : pos + 1, : pos + 1]
        scores = scores + rel_bias.unsqueeze(1)

        # Apply SiLU and normalize
        attn = F.silu(scores) / seq_len

        output = torch.matmul(attn, full_v)
        output = output.transpose(1, 2).contiguous().view(B, 1, D)

        return (
            output,
            full_k.transpose(1, 2).contiguous().view(B, -1, D),
            full_v.transpose(1, 2).contiguous().view(B, -1, D),
        )


class HSTUBlock(nn.Module):
    """HSTU block with gated attention and residual connection."""

    def __init__(
        self, embedding_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.max_seq_len = max_seq_len
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.uvqk_proj = nn.Linear(embedding_dim, embedding_dim * 4)
        self.attention = PointwiseAggregatedAttention(
            embedding_dim, num_heads, max_seq_len
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward. Args: x: [B, N, D]. Returns: [B, N, D]"""
        B, N, D = x.shape
        normed_x = self.input_norm(x)
        proj = self.uvqk_proj(normed_x)
        u, v, q, k = proj.chunk(4, dim=-1)
        u = F.silu(u)

        attn_out = self.attention(q, k, v, N)  # Pass actual seq_len
        attn_out = self.output_norm(attn_out)
        gated = u * attn_out

        # CRITICAL: Concatenate [u, attn_out, gated], not [u, x, gated]
        y = torch.cat([u, attn_out, gated], dim=-1)
        y = self.dropout(y)
        output = self.output_proj(y)

        # CRITICAL: Add residual connection
        return output + x

    def cached_forward(
        self, x: torch.Tensor, cache: Optional[Tuple], pos: int
    ) -> Tuple[torch.Tensor, Tuple]:
        """Incremental forward with cache. Args: x: [B, 1, D]. Returns: output [B, 1, D], new_cache"""
        normed_x = self.input_norm(x)
        proj = self.uvqk_proj(normed_x)
        u, v, q, k = proj.chunk(4, dim=-1)
        u = F.silu(u)

        cached_k, cached_v = cache if cache is not None else (None, None)
        attn_out, new_k, new_v = self.attention.cached_forward(
            q, k, v, cached_k, cached_v, pos, pos + 1
        )

        attn_out = self.output_norm(attn_out)
        gated = u * attn_out

        # CRITICAL: Concatenate [u, attn_out, gated], not [u, x, gated]
        y = torch.cat([u, attn_out, gated], dim=-1)
        y = self.dropout(y)
        output = self.output_proj(y)

        # CRITICAL: Add residual connection
        return output + x, (new_k, new_v)


class HSTU(DeepSequentialModel):
    """HSTU: Hierarchical Sequential Transduction Unit with incremental inference.

    From "Actions Speak Louder than Words: Trillion-Parameter Sequential
    Transducers for Generative Recommendations" (2024)

    Paper: https://arxiv.org/abs/2402.17152

    Key Features:
        - Pointwise aggregated attention with SiLU activation
        - Relative positional bias
        - Gated transformations with residual connections
        - **Incremental inference with KV-caching (true transduction)**
    """

    def __init__(self, config: HSTUConfig):
        super().__init__(config)
        self.save_hyperparameters()

        if config.position_config.get("type") == "rope":
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(config.max_seq_length, self.embedding_dim)
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

        self.hstu_blocks = nn.ModuleList(
            [
                HSTUBlock(
                    self.embedding_dim,
                    config.num_heads,
                    config.max_seq_length,
                    config.dropout_rate,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self._init_weights()

    def forward(self, sequences, sequence_lengths):
        """Forward pass through HSTU."""
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        logits = torch.matmul(hidden_states, self.item_embedding.weight.transpose(0, 1))
        return logits

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities."""
        logits = self.forward(sequences, sequence_lengths)
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=sequences.size(1) - 1
        )
        last_logits = logits[batch_indices, last_indices]
        return torch.softmax(last_logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings."""
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=hidden_states.size(1) - 1
        )
        return hidden_states[batch_indices, last_indices]

    def get_hidden_states(self, sequences, sequence_lengths):
        """Get hidden states from HSTU blocks."""
        batch_size, seq_len = sequences.size()

        item_embs = self.get_item_embedding(sequences)

        if self.pos_embedding is not None:
            positions = (
                torch.arange(seq_len, device=sequences.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            positions = torch.clamp(positions, 0, self.config.max_seq_length - 1)
            pos_embs = self.pos_embedding(positions)
            x = item_embs + pos_embs
        else:
            x = item_embs
        x = self.dropout(x)

        for block in self.hstu_blocks:
            x = block(x)  # Residual already inside block.forward()

        x = self.layer_norm(x)
        return x

    def incremental_forward(
        self, item_id: torch.Tensor, cache: Optional[List[List[Tuple]]] = None
    ) -> Tuple[torch.Tensor, List[List[Tuple]]]:
        """Incremental forward for one token. Args: item_id [B]. Returns: logits [B, vocab], new_cache"""
        x = self.get_item_embedding(item_id.unsqueeze(1))  # [B, 1, D]
        x = self.dropout(x)

        new_cache = []
        pos = 0 if cache is None else len(cache[0])

        for layer_idx, block in enumerate(self.hstu_blocks):
            layer_cache = cache[layer_idx] if cache is not None else None
            x, new_layer_cache = block.cached_forward(x, layer_cache, pos)
            new_cache.append(new_layer_cache)
            x = x + (cache[layer_idx] if cache is not None else x)  # Residual

        x = self.layer_norm(x)
        logits = torch.matmul(x, self.item_embedding.weight.transpose(0, 1))
        return logits.squeeze(1), new_cache

    def get_targets_and_mask(self, batch):
        """Get targets and loss mask for causal prediction."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        targets = sequences
        batch_size, seq_len = sequences.size()

        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        mask = (positions >= 1) & (positions < sequence_lengths.unsqueeze(1))

        return targets, mask

    def get_loss_mask(self, batch):
        """Get loss mask for negative sampling losses."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        batch_size, seq_len = sequences.size()

        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        mask = ((positions >= 1) & (positions < sequence_lengths.unsqueeze(1))).float()

        return mask

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.item_embedding.weight, std=0.02)

        for block in self.hstu_blocks:
            nn.init.xavier_uniform_(block.uvqk_proj.weight)
            nn.init.zeros_(block.uvqk_proj.bias)
            nn.init.xavier_uniform_(block.output_proj.weight)
            nn.init.zeros_(block.output_proj.bias)

            for module in block.modules():
                if isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
