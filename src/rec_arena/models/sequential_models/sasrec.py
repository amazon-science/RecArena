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

        # Position encoding. Supported types:
        #   "learnable" : additive learnable absolute embedding (SASRec default)
        #   "rope"      : rotary position embedding (order only)
        #   "to_rope"   : Time-and-Order RoPE (rotary over order + wall-clock time)
        #   "alibi"     : additive linear-bias attention (parameter-free)
        #   "t5_rab"    : learnable bucketed relative-position attention bias
        # The last two produce a per-head additive attention bias (self.pos_bias);
        # the rotary types produce self.rope; "learnable" produces pos_embedding.
        position_config = getattr(self.config, 'position_config', {"type": "learnable"})
        pos_type = position_config["type"]
        head_dim = self.embedding_dim // self.config.num_heads

        self.rope = None
        self.pos_embedding = None
        self.pos_bias = None
        self._uses_time = pos_type == "to_rope"

        if pos_type == "rope":
            from ...modules.layer_utils.embeddings import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(
                dim=head_dim,
                max_seq_len=self.config.max_seq_length,
                base=position_config.get("base", 10000),
            )
        elif pos_type == "to_rope":
            from ...modules.layer_utils.embeddings import TimeOrderRotaryEmbedding
            self.rope = TimeOrderRotaryEmbedding(
                dim=head_dim,
                max_seq_len=self.config.max_seq_length,
                base=position_config.get("base", 10000),
                time_ratio=position_config.get("time_ratio", 0.5),
                time_scale=position_config.get("time_scale", 1.0),
            )
        elif pos_type == "alibi":
            from ...modules.transformer_layers.position_bias import ALiBiBias
            self.pos_bias = ALiBiBias(
                num_heads=self.config.num_heads,
                max_seq_len=self.config.max_seq_length,
            )
        elif pos_type == "t5_rab":
            from ...modules.transformer_layers.position_bias import (
                T5RelativeAttentionBias,
            )
            self.pos_bias = T5RelativeAttentionBias(
                num_heads=self.config.num_heads,
                max_seq_len=self.config.max_seq_length,
                num_buckets=position_config.get("num_buckets", 32),
                max_distance=position_config.get("max_distance", 128),
            )
        else:  # "learnable"
            self.pos_embedding = nn.Embedding(
                self.config.max_seq_length, self.embedding_dim
            )
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

        activation = self._get_activation()
        qk_norm = getattr(self.config, "use_qk_norm", False)
        peri_norm = getattr(self.config, "use_peri_norm", False)

        norm_eps = getattr(self.config, "layer_norm_eps", 1e-5)
        use_bias = getattr(self.config, "use_bias", False)

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
                    qk_norm=qk_norm,
                    pos_bias=self.pos_bias,
                    peri_norm=peri_norm,
                    bias=use_bias,
                    norm_eps=norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Optional LayerNorm on the input embeddings (RecBole-faithful mode).
        # Default off: RecArena has historically applied no input LN.
        self.input_layer_norm = (
            nn.LayerNorm(self.embedding_dim, eps=norm_eps)
            if getattr(self.config, "input_layer_norm", False)
            else None
        )
        # Final layer norm after all transformer blocks. Suppressible so a run
        # can match RecBole SASRec (which has none). Default on (current behavior).
        self.layer_norm = (
            nn.LayerNorm(self.embedding_dim, eps=norm_eps)
            if getattr(self.config, "final_layer_norm", True)
            else None
        )
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
            # RoPE / TO-RoPE / additive-bias position encodings act inside the
            # attention layers, so no additive input embedding here.
            x = item_embs

        # RecBole-faithful: LayerNorm the input embeddings before dropout.
        if self.input_layer_norm is not None:
            x = self.input_layer_norm(x)

        x = self.dropout(x)

        # TO-RoPE consumes real interaction timestamps (falls back to positions
        # when unavailable). Other position types ignore this.
        timestamps = None
        if self._uses_time:
            timestamps = self.get_batch_timestamps()
            if timestamps is not None:
                timestamps = timestamps.to(sequences.device).float()

        # CRITICAL: Don't pass padding mask when using causal attention
        # The is_causal flag is ignored when attn_mask is provided!
        # Apply transformer blocks with causal attention only
        for block in self.transformer_blocks:
            x = block(x, attn_mask=None, timestamps=timestamps)

        # Apply final layer norm before output projection (suppressible for
        # RecBole-faithful mode, which has no post-encoder LayerNorm).
        if self.layer_norm is not None:
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
