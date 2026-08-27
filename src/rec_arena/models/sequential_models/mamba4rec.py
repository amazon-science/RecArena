import torch
import torch.nn as nn
from ..sequential import DeepSequentialModel
from ...configs.defaults.mamba4rec import Mamba4RecConfig
from ...modules.mamba_utils.mamba import ResidualBlock


class Mamba4Rec(DeepSequentialModel):
    """Mamba-based sequential recommendation model."""

    def __init__(self, config: Mamba4RecConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # Mamba layers
        self.mamba_layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand_factor=config.expand_factor,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init_floor=config.dt_init_floor,
                    conv_bias=config.conv_bias,
                    bias=config.bias,
                    norm=config.norm,
                    layer_idx=i,
                    mamba_version=config.mamba_version,
                )
                for i in range(config.num_layers)
            ]
        )

        # Project embedding to Mamba dimension if different
        if self.embedding_dim != config.d_model:
            self.input_projection = nn.Linear(self.embedding_dim, config.d_model)
        else:
            self.input_projection = nn.Identity()

        # Project Mamba output back to embedding dimension for logit computation
        if config.d_model != self.embedding_dim:
            self.output_projection = nn.Linear(config.d_model, self.embedding_dim)
        else:
            self.output_projection = nn.Identity()

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, sequences, sequence_lengths):
        """Forward pass through Mamba4Rec - returns logits for ALL positions."""
        # Get all hidden states: [batch_size, seq_len, d_model]
        # NOTE: dropout is applied INSIDE get_hidden_states so it is shared by
        # both the eval (forward) and train (base compute_loss -> hidden_states)
        # paths. Do NOT re-apply it here or eval would double-drop.
        all_hidden = self.get_hidden_states(sequences, sequence_lengths)

        # Project back to embedding dimension
        all_hidden = self.output_projection(all_hidden)

        # Compute logits for ALL positions
        logits = torch.matmul(all_hidden, self.item_embedding.weight.transpose(0, 1))

        return logits  # [batch_size, seq_len, vocab_size]

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities."""
        logits = self.forward(sequences, sequence_lengths)  # [batch, seq_len, vocab]
        
        # Extract last valid position
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        last_logits = logits[batch_indices, last_indices]
        
        return torch.softmax(last_logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings - return last hidden state."""
        all_hidden = self.get_hidden_states(sequences, sequence_lengths)
        
        # Extract last valid position
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        
        return all_hidden[batch_indices, last_indices]

    def get_hidden_states(self, sequences, sequence_lengths):
        """Get hidden states from Mamba layers - returns ALL positions."""
        # Get item embeddings [batch_size, seq_len, embedding_dim]
        embedded = self.get_item_embedding(sequences)

        # Project to Mamba dimension [batch_size, seq_len, d_model]
        x = self.input_projection(embedded)

        # Pass through Mamba layers
        for layer in self.mamba_layers:
            x = layer(x)

        # Dropout on the Mamba output. Applied here (not only in forward) so the
        # configured dropout_rate is active during training too -- training goes
        # through the base compute_loss -> get_hidden_states, which previously
        # applied NO dropout, leaving dropout_rate inert during training.
        x = self.dropout(x)

        # Return all positions for per-position loss computation
        return x  # [batch_size, seq_len, d_model]

    def get_targets_and_mask(self, batch):
        """Get targets and loss mask for Mamba4Rec causal prediction."""
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
