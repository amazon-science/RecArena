import torch
import torch.nn as nn
from ..sequential import DeepSequentialModel
from ...configs.defaults.gru4rec import GRU4RecConfig


class GRU4Rec(DeepSequentialModel):
    """GRU4Rec: Session-based Recommendations with Recurrent Neural Networks.

    A GRU-based model for session-based recommendation that captures sequential
    patterns using recurrent neural networks.

    Paper: "Session-based Recommendations with Recurrent Neural Networks" (ICLR 2016)
    Link: https://arxiv.org/abs/1511.06939

    Model ID: gru4rec
    Model Type: Sequential

    Key Features:
        - GRU layers for sequential modeling
        - Efficient for long sequences
        - Packed sequences for variable lengths
        - Dropout for regularization

    Args:
        config (GRU4RecConfig): Model configuration with GRU parameters

    Example:
        >>> config = GRU4RecConfig(vocab_size=1000, hidden_size=64, num_layers=1)
        >>> model = GRU4Rec(config)
        >>> logits = model.forward(sequences, sequence_lengths)
    """

    def __init__(self, config: GRU4RecConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # GRU layer
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout_rate if self.config.num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(self.config.dropout_rate)

    def forward(self, sequences, sequence_lengths):
        """Forward pass through GRU4Rec - returns logits for ALL positions."""
        # Get all hidden states: [batch_size, seq_len, hidden_size]
        all_hidden = self.get_hidden_states(sequences, sequence_lengths)
        all_hidden = self.dropout(all_hidden)

        # Compute logits for ALL positions (needed for per-position loss)
        logits = torch.matmul(all_hidden, self.item_embedding.weight.transpose(0, 1))

        return logits  # [batch_size, seq_len, vocab_size]

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities."""
        logits = self.forward(sequences, sequence_lengths)  # [batch, seq_len, vocab]
        
        # Extract last valid position
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        last_logits = logits[batch_indices, last_indices]  # [batch, vocab]
        
        return torch.softmax(last_logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings - return last hidden state."""
        all_hidden = self.get_hidden_states(sequences, sequence_lengths)

        # Get last valid hidden state
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)

        return all_hidden[batch_indices, last_indices]

    def get_hidden_states(self, sequences, sequence_lengths):
        """Get hidden states from GRU - return all hidden states for loss computation."""
        # Get item embeddings
        embedded = self.get_item_embedding(sequences)

        # Pack sequences for efficient processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # GRU forward pass
        packed_output, _ = self.gru(packed)

        # Unpack sequences - use total_length to ensure same shape as input
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=sequences.size(1)
        )

        return output  # [batch_size, seq_len, hidden_size]

    def get_targets_and_mask(self, batch):
        """Get targets and mask for causal prediction."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        # Targets are the sequences themselves (will be shifted in compute_loss)
        targets = sequences

        # Create causal mask: predict position i from positions 0..i-1
        batch_size, seq_len = sequences.size()
        mask = torch.zeros(batch_size, seq_len, device=sequences.device, dtype=torch.bool)
        for i, length in enumerate(sequence_lengths):
            if length > 1:
                mask[i, 1:length] = True  # Predict positions 1 to length-1

        return targets, mask
