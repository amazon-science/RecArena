"""Caser: Convolutional Sequence Embedding Recommendation Model."""

import torch
import torch.nn as nn
from ..sequential import DeepSequentialModel
from ...configs.defaults.caser import CaserConfig


class Caser(DeepSequentialModel):
    """Caser: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding.

    A CNN-based model that uses horizontal and vertical convolutional filters to capture
    both union-level and point-level sequential patterns.

    Paper: "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" (WSDM 2018)
    Link: https://arxiv.org/abs/1809.07426

    Model ID: caser
    Model Type: Sequential

    Key Features:
        - Horizontal convolutions for union-level patterns (n-grams)
        - Vertical convolutions for point-level patterns
        - Multiple filter sizes for diverse pattern capture
        - Configurable activation functions

    Args:
        config (CaserConfig): Model configuration with CNN parameters

    Example:
        >>> config = CaserConfig(vocab_size=1000, horizontal_filter_sizes=[2, 3, 4])
        >>> model = Caser(config)
        >>> logits = model.forward(sequences)
    """

    # compute_loss forms full-vocab logits (dense table gradient) -> no sparse.
    SUPPORTS_SPARSE = False

    def __init__(self, config: CaserConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # Horizontal convolution filters (capture union-level patterns)
        self.horizontal_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    self.config.num_horizontal_filters,
                    (filter_size, self.config.embedding_dim),
                )
                for filter_size in self.config.horizontal_filter_sizes
            ]
        )

        # Vertical convolution filter (capture point-level patterns)
        self.vertical_conv = nn.Conv2d(
            1, self.config.num_vertical_filters, (self.config.vertical_filter_size, 1)
        )

        # Fully connected layer
        num_horizontal_out = (
            len(self.config.horizontal_filter_sizes)
            * self.config.num_horizontal_filters
        )
        num_vertical_out = self.config.num_vertical_filters * self.config.embedding_dim
        fc_input_dim = num_horizontal_out + num_vertical_out

        self.fc = nn.Linear(fc_input_dim, self.config.embedding_dim)

        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        self.dropout = nn.Dropout(self.config.dropout_rate)

        # Activation function
        self.activation = self._get_activation()

        # No output layer needed - use item_embedding.weight for logits

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Parent class already initializes item_embedding
        for conv in self.horizontal_convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.vertical_conv.weight)
        nn.init.zeros_(self.vertical_conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, sequence, sequence_length=None):
        """Forward pass through Caser."""
        # Get hidden states
        hidden = self.get_hidden_states(sequence, sequence_length)

        # Compute logits same way as training
        logits = torch.matmul(hidden, self.get_output_embeddings().transpose(0, 1))

        return logits

    def get_hidden_states(self, sequence, sequence_length=None):
        """Get hidden representations for loss computation."""
        # Get embeddings using parent's method
        item_embs = self.get_item_embedding(sequence).unsqueeze(1)

        # Horizontal convolution
        horizontal_out = []
        for conv in self.horizontal_convs:
            conv_out = self.activation(conv(item_embs))
            conv_out = conv_out.squeeze(3)
            pool_out = torch.max_pool1d(conv_out, conv_out.size(2))
            horizontal_out.append(pool_out.squeeze(2))
        horizontal_out = torch.cat(horizontal_out, dim=1)

        # Vertical convolution
        vertical_out = self.activation(self.vertical_conv(item_embs))
        vertical_out = vertical_out.squeeze(2).view(vertical_out.size(0), -1)

        # Combine
        combined = torch.cat([horizontal_out, vertical_out], dim=1)
        combined = self.dropout(combined)
        hidden = self.activation(self.fc(combined))

        x = self.layer_norm(hidden)

        return x

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities."""
        logits = self.forward(sequences, sequence_lengths)
        return torch.softmax(logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings."""
        return self.get_hidden_states(sequences, sequence_lengths)

    def compute_loss(self, batch):
        """Compute loss for Caser.

        Unlike other sequential models, Caser predicts only the NEXT item
        from the entire sequence, not per-position predictions.
        """
        if self.loss_fn is None:
            raise RuntimeError(
                "No loss function set. Either pass loss_fn in config or use external loss functions."
            )

        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        # Get the target (next item after each sequence)
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        targets = sequences[batch_indices, last_indices]  # [batch_size]

        # Remove last item from sequences for prediction
        pred_sequences = sequences.clone()
        for i, length in enumerate(sequence_lengths):
            if length > 0:
                pred_sequences[i, length - 1] = 0

        # Get predictions (2D: [batch_size, vocab_size])
        logits = self.forward(pred_sequences, sequence_lengths)

        # Create mask for valid predictions (exclude padding and special tokens)
        mask = targets >= 3  # Items start at 3

        # Reshape for loss function compatibility
        # Loss functions expect: (batch_size, vocab_size) or (batch_size, seq_len, vocab_size)
        logits = logits.unsqueeze(1)  # [batch_size, 1, vocab_size]
        targets = targets.unsqueeze(1)  # [batch_size, 1]
        mask = mask.unsqueeze(1)  # [batch_size, 1]

        # Extract negatives if present
        neg_items = batch.get("neg_items")

        return self.loss_fn(
            logits=logits,
            targets=targets,
            mask=mask,
            neg_items=neg_items,
        )

    def get_targets_and_mask(self, batch):
        """Get targets and loss mask for Caser.

        Note: This method is not used since compute_loss is overridden.
        Kept for compatibility.
        """
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        # Get next item target
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        targets = sequences[batch_indices, last_indices]

        # Create mask
        mask = targets > 0

        return targets.unsqueeze(1), mask.unsqueeze(1)
