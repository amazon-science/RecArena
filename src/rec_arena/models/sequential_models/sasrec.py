import torch
import torch.nn as nn
from ...modules.transformer_layers.transformer_block import TransformerBlock
from ..sequential import DeepSequentialModel
from ...configs.defaults.sasrec import SASRecConfig


class SASRec(DeepSequentialModel):
    """SASRec: Self-Attentive Sequential Recommendation.

    A Transformer-based sequential recommendation model that uses causal self-attention
    to capture sequential patterns in user behavior.

    Paper: "Self-Attentive Sequential Recommendation" (KDD 2018)
    Link: https://arxiv.org/abs/1808.09781

    Model ID: sasrec
    Model Type: Sequential

    Key Features:
        - Causal self-attention mechanism
        - Position embeddings for sequence order
        - Multi-head attention for diverse patterns
        - Layer normalization and dropout for stability

    Args:
        config (SASRecConfig): Model configuration with architecture parameters

    Example:
        >>> config = SASRecConfig(vocab_size=1000, embedding_dim=64, num_layers=2)
        >>> model = SASRec(config)
        >>> logits = model.forward(sequences, sequence_lengths)
    """

    def __init__(self, config: SASRecConfig):
        super().__init__(config)
        self.save_hyperparameters()

        # Position encoding (learnable or RoPE)
        position_config = getattr(self.config, 'position_config', {"type": "learnable"})
        if position_config["type"] == "rope":
            from ...modules.layer_utils.embeddings import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(
                dim=self.embedding_dim // self.config.num_heads,
                max_seq_len=self.config.max_seq_length,
                base=position_config.get("base", 10000)
            )
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(
                self.config.max_seq_length, self.embedding_dim
            )
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
            self.rope = None

        activation = self._get_activation()

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embedding_dim,
                    num_heads=self.config.num_heads,
                    hidden_dim=self.config.feedforward_dim,
                    dropout_rate=self.config.dropout_rate,
                    activation=activation,
                    use_swiglu=self.config.use_ligr,
                    use_rms_norm=getattr(self.config, 'use_rms_norm', False),
                    use_gated_residual=self.config.use_ligr,
                    norm_first=self.config.layer_norm_first,
                    rope=self.rope,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Final layer norm (applied after all transformer blocks)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        # Input dropout (applied to embeddings)
        self.dropout = nn.Dropout(self.config.dropout_rate)

        # Output layer configuration
        self.tie_embeddings = getattr(self.config, 'tie_embeddings', True)
        self.output_lora_rank = getattr(self.config, 'output_lora_rank', 0)
        
        if not self.tie_embeddings:
            # Separate output embedding matrix
            self.output_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        elif self.output_lora_rank > 0:
            # LoRA adapter for output: W_out = W_emb + A @ B
            self.lora_A = nn.Parameter(torch.zeros(self.embedding_dim, self.output_lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(self.output_lora_rank, self.vocab_size))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            # B initialized to zero so LoRA starts as identity

        # Initialize weights properly
        self._init_weights()

    def forward(self, sequences, sequence_lengths):
        """Forward pass through SASRec.

        Args:
            sequences (torch.Tensor): Input sequences [batch_size, seq_len]
            sequence_lengths (torch.Tensor): Actual lengths [batch_size]

        Returns:
            torch.Tensor: Logits for all positions [batch_size, seq_len, vocab_size]
        """
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        logits = self._compute_logits(hidden_states)
        return logits
    
    def _compute_logits(self, hidden_states):
        """Compute logits from hidden states using appropriate output layer."""
        return torch.matmul(hidden_states, self.get_output_embeddings().transpose(0, 1))

    def get_output_embeddings(self) -> torch.Tensor:
        """Get output embedding weights for logit computation."""
        if not self.tie_embeddings:
            return self.output_embedding.weight
        elif self.output_lora_rank > 0:
            # Tied + LoRA: W_out = W_emb + (A @ B).T
            # Return [vocab_size, embedding_dim]
            lora_delta = torch.matmul(self.lora_A, self.lora_B).transpose(0, 1)  # [vocab, emb]
            return self.item_embedding.weight + lora_delta
        else:
            return self.item_embedding.weight

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities.

        Args:
            sequences (torch.Tensor): Input sequences [batch_size, seq_len]
            sequence_lengths (torch.Tensor): Actual lengths [batch_size]

        Returns:
            torch.Tensor: Probability distribution over items [batch_size, vocab_size]
        """
        logits = self.forward(sequences, sequence_lengths)

        # Get last valid position
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=sequences.size(1) - 1
        )
        last_logits = logits[batch_indices, last_indices]

        return torch.softmax(last_logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings - reuse get_hidden_states."""
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)

        # Extract last valid position
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=hidden_states.size(1) - 1
        )
        return hidden_states[batch_indices, last_indices]

    def get_hidden_states(self, sequences, sequence_lengths):
        """Get hidden states from transformer - core SASRec computation."""
        batch_size, seq_len = sequences.size()

        # Get item embeddings
        item_embs = self.get_item_embedding(sequences)

        # Scale embeddings by sqrt(d) BEFORE adding position embeddings
        if self.config.scale_embeddings:
            item_embs = item_embs * (self.embedding_dim ** 0.5)
        
        # Add position embeddings (if using learnable)
        if self.pos_embedding is not None:
            positions = torch.clamp(
                torch.arange(seq_len, device=sequences.device)
                .unsqueeze(0)
                .repeat(batch_size, 1),
                0,
                self.config.max_seq_length - 1,
            )
            pos_embs = self.pos_embedding(positions)
            x = item_embs + pos_embs
        else:
            # RoPE is applied in attention layers
            x = item_embs
        
        x = self.dropout(x)

        # CRITICAL: Don't pass padding mask when using causal attention
        # The is_causal flag is ignored when attn_mask is provided!
        # Apply transformer blocks with causal attention only
        for block in self.transformer_blocks:
            x = block(x, attn_mask=None)

        # Apply final layer norm before output projection
        x = self.layer_norm(x)

        return x

    def get_targets_and_mask(self, batch):
        """Get targets and loss mask for SASRec causal prediction."""
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
            if length > 0:
                mask[i, 1:length] = True  # Predict positions 1 to length-1

        return targets, mask

    def get_loss_mask(self, batch):
        """Get loss mask for negative sampling losses."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        batch_size, seq_len = sequences.size()
        mask = torch.zeros(batch_size, seq_len, device=sequences.device)

        for i, length in enumerate(sequence_lengths):
            if length > 0:
                mask[i, 1:length] = 1.0  # Predict positions 1 to length-1

        return mask

    def _init_weights(self):
        """Initialize weights to prevent NaN."""
        # Initialize item embeddings
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
        if hasattr(self, 'output_embedding'):
            nn.init.normal_(self.output_embedding.weight, std=0.02)


#
