"""FMLP-Rec: Filter-enhanced MLP for Sequential Recommendation."""

import torch
import torch.nn as nn
import torch.fft as fft
from ..sequential import DeepSequentialModel
from ...configs.defaults.fmlprec import FMLPRecConfig


class FilterLayer(nn.Module):
    """Filter layer matching original FMLP-Rec implementation."""

    def __init__(self, seq_len, embedding_dim, dropout_rate):
        super().__init__()
        self.seq_len = seq_len
        
        # Exact match to original: 0.02 initialization
        self.complex_weight = nn.Parameter(
            torch.randn(1, seq_len // 2 + 1, embedding_dim, 2, dtype=torch.float32) * 0.02
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)

    def forward(self, input_tensor):
        """FilterLayer with causal masking - processes each position with its causal prefix."""
        batch, seq_len, hidden = input_tensor.shape
        
        # Create all causal prefixes efficiently using broadcasting
        # Create mask: [S, S] where mask[i, j] = 1 if j <= i else 0
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_tensor.device, dtype=input_tensor.dtype))
        
        # Broadcast and apply: [B, S, H] -> [B, S, S, H] -> [B*S, S, H]
        causal_batch = (input_tensor.unsqueeze(1) * causal_mask.unsqueeze(0).unsqueeze(-1)).view(batch * seq_len, seq_len, hidden)
        
        # Single FFT call on all causal prefixes
        x = fft.rfft(causal_batch, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        
        # Reshape: [S, B, S, H] and extract diagonal (position i from prefix i)
        sequence_emb_fft = sequence_emb_fft.view(seq_len, batch, seq_len, hidden)
        hidden_states = torch.stack([sequence_emb_fft[i, :, i, :] for i in range(seq_len)], dim=1)
        
        # Dropout and residual
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        
        return hidden_states


class Intermediate(nn.Module):
    """MLP layer matching original FMLP-Rec implementation."""

    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        # Original uses 4x expansion
        self.dense_1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.dense_2 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tensor):
        """Exact match to original Intermediate implementation."""
        hidden_states = self.dense_1(input_tensor)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Single residual connection
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        
        return hidden_states


class FMLPLayer(nn.Module):
    """Combined layer matching original Layer implementation."""

    def __init__(self, seq_len, embedding_dim, dropout_rate):
        super().__init__()
        self.filter_layer = FilterLayer(seq_len, embedding_dim, dropout_rate)
        self.intermediate = Intermediate(embedding_dim, dropout_rate)

    def forward(self, hidden_states, attention_mask=None):
        """Exact match to original Layer.forward()."""
        # Filter first (FilterLayer ignores attention_mask like original)
        hidden_states = self.filter_layer(hidden_states)
        # Then MLP
        hidden_states = self.intermediate(hidden_states)
        return hidden_states


class FMLPRec(DeepSequentialModel):
    """FMLPRec: Filter-enhanced MLP for Sequential Recommendation.

    An MLP-only model that uses learnable filters in the frequency domain (via FFT)
    to capture sequential patterns without attention or convolution.

    Paper: "Filter-enhanced MLP is All You Need for Sequential Recommendation" (WWW 2022)
    Link: https://arxiv.org/abs/2211.14582

    Model ID: fmlprec
    Model Type: Sequential

    Key Features:
        - MLP-only architecture (no attention/convolution)
        - Learnable complex filters in frequency domain
        - FFT/IFFT for efficient filtering
        - Tunable MLP hidden dimensions

    Args:
        config (FMLPRecConfig): Model configuration with MLP and filter parameters

    Example:
        >>> config = FMLPRecConfig(vocab_size=1000, embedding_dim=64, mlp_hidden_dim=256)
        >>> model = FMLPRec(config)
        >>> logits = model.forward(sequences, sequence_lengths)
    """

    def __init__(self, config: FMLPRecConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # FMLP layers matching original architecture
        self.blocks = nn.ModuleList(
            [
                FMLPLayer(
                    self.config.max_seq_length,
                    self.config.embedding_dim,
                    self.config.dropout_rate,
                )
                for _ in range(self.config.num_blocks)
            ]
        )

        # FMLPRec requires learnable position embeddings for frequency filters
        position_config = getattr(self.config, 'position_config', {"type": "learnable"})
        if position_config["type"] == "rope":
            raise ValueError(
                "FMLPRec requires learnable position embeddings. "
                "RoPE is not supported as it's designed for attention mechanisms."
            )
        self.pos_embedding = nn.Embedding(
            self.config.max_seq_length, self.embedding_dim
        )
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # LayerNorm for embedding combination (like original)
        self.layer_norm = nn.LayerNorm(self.embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights matching original implementation."""
        # Original uses 0.02 std for all embeddings
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize layer norm
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
        
        # Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sequence, sequence_length=None):
        """Forward pass through FMLP-Rec."""
        # Get hidden states
        hidden = self.get_hidden_states(sequence, sequence_length)

        # Compute logits same way as training
        logits = torch.matmul(hidden, self.item_embedding.weight.transpose(0, 1))

        return logits

    def get_hidden_states(self, sequences, sequence_lengths):
        """Get hidden states from FMLP blocks with proper causal masking."""
        batch_size, seq_len = sequences.size()

        # Simple padding mask (FilterLayer doesn't use attention_mask anyway)
        attention_mask = (sequences > 0).long()  # Padding mask

        # Get item embeddings
        item_embs = self.get_item_embedding(sequences)

        # Position embeddings (always required for FMLPRec)
        positions = torch.clamp(
            torch.arange(seq_len, device=sequences.device)
            .unsqueeze(0)
            .repeat(batch_size, 1),
            0,
            self.config.max_seq_length - 1,
        )
        pos_embs = self.pos_embedding(positions)
        x = item_embs + pos_embs
        
        # CRITICAL: Apply LayerNorm + Dropout after embedding combination (like original)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Apply FMLP blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        return x

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities for the next position after each sequence.

        Args:
            sequences: [batch_size, seq_len] input sequences
            sequence_lengths: [batch_size] actual sequence lengths

        Returns:
            [batch_size, vocab_size] probability distribution over next items
        """
        # Get logits for all positions
        logits = self.forward(
            sequences, sequence_lengths
        )  # [batch_size, seq_len, vocab_size]

        # Extract logits at last valid position for each sequence
        batch_size = sequences.size(0)
        batch_indices = torch.arange(batch_size, device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)

        # Get logits at last position: [batch_size, vocab_size]
        last_logits = logits[batch_indices, last_indices]

        return torch.softmax(last_logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings."""
        return self.get_hidden_states(sequences, sequence_lengths)

    def get_targets_and_mask(self, batch):
        """Get targets and loss mask for FMLPRec causal prediction."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        # Targets are the input sequence (causal masking handles the shift)
        targets = sequences

        # Create causal mask: predict position i from positions 0..i-1
        batch_size, seq_len = sequences.size()
        mask = torch.zeros(
            batch_size, seq_len, device=sequences.device, dtype=torch.bool
        )

        for i, length in enumerate(sequence_lengths):
            if length > 1:
                mask[i, 1:length] = True  # Predict positions 1 to length-1

        return targets, mask
