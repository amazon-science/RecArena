import torch
import torch.nn as nn
import math
from ..sequential import DeepSequentialModel
from ...configs.defaults.bert4rec import BERT4RecConfig
from ...modules.transformer_layers.transformer_block import TransformerBlock


class BERT4Rec(DeepSequentialModel):
    """BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations.

    A bidirectional Transformer model that uses masked language modeling (MLM) to learn
    from both past and future context in user sequences.

    Paper: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
            from Transformer" (CIKM 2019)
    Link: https://arxiv.org/abs/1904.06690

    Model ID: bert4rec
    Model Type: Sequential

    Key Features:
        - Bidirectional self-attention (no causal masking)
        - Masked language modeling training objective
        - Cloze task for learning item representations
        - Configurable masking strategy (80/10/10 by default)

    Args:
        config (BERT4RecConfig): Model configuration with architecture and masking parameters

    Example:
        >>> config = BERT4RecConfig(vocab_size=1000, embedding_dim=64, mask_prob=0.15)
        >>> model = BERT4Rec(config)
        >>> logits = model.forward(sequences)
    """

    def __init__(self, config: BERT4RecConfig):
        super().__init__(config)

        self.num_heads = config.get("num_heads", 2)
        self.num_layers = config.get("num_layers", 2)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        self.hidden_units = config.get("hidden_units", self.embedding_dim)
        self.mask_prob = config.get("mask_prob", 0.2)

        # CRITICAL FIX: MASK token is at index 2 (GPT-style)
        # Items are [3, 4, 5, ...] so no conflict
        self.mask_token_id = 2

        # Position embedding
        position_config = getattr(config, 'position_config', {"type": "learnable"})
        if position_config["type"] == "rope":
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(self.max_seq_length, self.embedding_dim)

        activation_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        activation = activation_map.get(
            self.config.transformer_activation.lower(), nn.GELU()
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embedding_dim,
                    num_heads=self.config.num_heads,
                    hidden_dim=self.embedding_dim * 4,
                    dropout_rate=self.config.dropout_rate,
                    activation=activation,
                    causality=False,
                    use_swiglu=self.config.use_ligr,  # Use SwiGLU when LiGR mode enabled
                    use_gated_residual=self.config.use_ligr,  # Use gated residuals when LiGR mode enabled
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        # Initialize weights properly (CRITICAL for convergence!)
        self._init_weights()

    def forward(self, sequences, attention_mask=None):
        """Forward pass through BERT4Rec."""
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = sequences != 0

        # Get hidden states with attention mask
        hidden_states = self.get_hidden_states(sequences, attention_mask=attention_mask)

        # Project to vocabulary for masked prediction (use tied weights like SASRec)
        logits = torch.matmul(hidden_states, self.item_embedding.weight.transpose(0, 1))

        return logits

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities."""
        # For inference, mask the last position and predict
        masked_sequences = sequences.clone()
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        masked_sequences[batch_indices, last_indices] = self.mask_token_id

        # Create attention mask from masked sequences (not original)
        # CRITICAL: mask_token_id is a valid token, should not be masked out
        attention_mask = masked_sequences != 0
        logits = self.forward(masked_sequences, attention_mask=attention_mask)

        # Get predictions for the last position
        last_logits = logits[batch_indices, last_indices]  # [batch_size, vocab_size]

        # Remove padding token and mask token from predictions
        last_logits[:, 0] = -float("inf")  # PAD
        last_logits[:, 1] = -float("inf")  # UNK
        last_logits[:, 2] = -float("inf")  # MASK

        return torch.softmax(last_logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings - reuse get_hidden_states with mean pooling."""
        attention_mask = sequences != 0
        hidden_states = self.get_hidden_states(sequences, attention_mask=attention_mask)

        # Mean pooling over valid positions
        seq_len = hidden_states.size(1)
        mask = torch.arange(seq_len, device=sequences.device).unsqueeze(
            0
        ) < sequence_lengths.unsqueeze(1)
        mask = mask.float().unsqueeze(-1)  # [batch_size, seq_len, 1]

        masked_embeddings = hidden_states * mask
        sequence_embeddings = (
            masked_embeddings.sum(dim=1) / sequence_lengths.unsqueeze(-1).float()
        )

        return sequence_embeddings

    def mask_sequences(self, sequences, mask_prob=None):
        """Apply BERT-style masking to sequences for external loss functions."""
        if mask_prob is None:
            mask_prob = self.config.mask_prob

        masked_sequences = sequences.clone()
        labels = torch.full_like(sequences, -100)  # -100 is ignored in loss

        # Create random mask (don't mask padding tokens)
        mask_prob_matrix = torch.rand(sequences.shape, device=sequences.device)
        padding_mask = sequences == 0  # PAD_TOKEN = 0
        # Set padding positions to 1.0 so they're never selected (since we check < mask_prob)
        mask_prob_matrix.masked_fill_(padding_mask, 1.0)

        masked_indices = mask_prob_matrix < mask_prob
        # Clamp labels to valid vocab range
        valid_tokens = torch.clamp(sequences[masked_indices], 0, self.vocab_size - 1)
        labels[masked_indices] = valid_tokens

        # mask_token_prob% of the time, replace with [MASK] token
        indices_replaced = (
            torch.bernoulli(
                torch.full(
                    labels.shape, self.config.mask_token_prob, device=sequences.device
                )
            ).bool()
            & masked_indices
        )
        masked_sequences[indices_replaced] = self.mask_token_id

        # random_token_prob% of the time, replace with random token
        # Adjust probability: random_token_prob / (1 - mask_token_prob)
        random_prob = (
            self.config.random_token_prob / (1 - self.config.mask_token_prob)
            if self.config.mask_token_prob < 1
            else 0
        )
        indices_random = (
            torch.bernoulli(
                torch.full(labels.shape, random_prob, device=sequences.device)
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        if self.vocab_size > 3:
            # Random tokens should be valid items [3, 4, ..., vocab_size-1]
            # Exclude: 0 (PAD), 1 (UNK), 2 (MASK)
            random_tokens = torch.randint(
                3, self.vocab_size, labels.shape, device=sequences.device
            )
            masked_sequences[indices_random] = random_tokens[indices_random]

        return masked_sequences, labels

    def get_hidden_states(self, sequences, sequence_lengths=None, attention_mask=None):
        """Get hidden states from BERT transformer.
        
        Args:
            sequences: [batch, seq_len] input sequences
            sequence_lengths: [batch] lengths (optional, for compatibility)
            attention_mask: [batch, seq_len] attention mask (optional)
        """
        batch_size, seq_len = sequences.size()

        # Get item embeddings directly - no offset needed with mask_token_id at vocab_size-1
        item_embs = self.item_embedding(sequences)

        # Scale embeddings by sqrt(d) BEFORE adding position embeddings
        if getattr(self.config, 'scale_embeddings', True):
            item_embs = item_embs * (self.embedding_dim ** 0.5)

        # Position embeddings (clamp to max_seq_length)
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
            x = item_embs
        x = self.dropout(x)

        # Create attention mask if not provided (True for valid positions, False for padding)
        if attention_mask is None:
            attention_mask = sequences != 0  # [batch_size, seq_len] - keep as bool
        
        # Ensure mask is boolean (don't convert to float - MHA module handles that)
        if attention_mask.dtype not in [torch.bool]:
            attention_mask = attention_mask.bool()

        # Apply transformer blocks (bidirectional attention with padding mask)
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attention_mask)

        x = self.layer_norm(x)
        return x

    def _init_weights(self):
        """Initialize weights to prevent NaN and ensure proper convergence."""
        # Initialize item embeddings with small std like SASRec
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        # Initialize position embeddings with small std
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def compute_loss(self, batch):
        """Compute loss for BERT4Rec - uses masked prediction (not causal).

        BERT4Rec is different from causal models:
        - Predicts masked positions in-place (not next item)
        - No causal shifting needed
        - Uses bidirectional attention
        """
        if self.loss_fn is None:
            raise RuntimeError("No loss function set")

        sequences = batch["sequence"]

        # Apply masking for BERT4Rec
        masked_sequences, labels = self.mask_sequences(sequences)

        # Create attention mask for valid (non-padding) positions
        attention_mask = sequences != 0

        # Get hidden states with masked input and attention mask
        hidden_states = self.get_hidden_states(
            masked_sequences, attention_mask=attention_mask
        )

        # Create mask: only compute loss for masked positions
        # labels has -100 for non-masked positions, valid item IDs for masked positions
        mask = (labels != -100).float()

        # Padding (0) should be ignored, keep -100 as -100 for ignore_index
        # Valid targets stay 1-indexed to match logits
        # CRITICAL: Clamp targets to valid vocab range to prevent CUDA assertion errors
        targets = torch.where(
            labels == -100, 
            torch.tensor(-100, device=labels.device), 
            torch.clamp(labels, 0, self.vocab_size - 1)
        )

        # Extract negative samples if present
        neg_items = batch.get("neg_items")
        # Note: neg_items don't need shifting for BERT4Rec (in-place prediction)

        # Call loss function with hidden states for fast sampled logits
        # Loss functions will use fast path if they support it
        return self.loss_fn(
            hidden_states=hidden_states,
            item_embeddings=self.item_embedding.weight,
            targets=targets,
            mask=mask,
            neg_items=neg_items,
        )
