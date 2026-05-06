"""MLP4Rec: Simple MLP baseline for sequential recommendation."""

import torch
import torch.nn as nn
from ..sequential import DeepSequentialModel
from ...configs.defaults.mlp4rec import MLP4RecConfig
from ...modules.layer_utils.batch_ensembling import LinearBatchEnsembleLayer


class MLP4Rec(DeepSequentialModel):
    """MLP4Rec: Pure MLP for Sequential Recommendation.

    A simple MLP-based model that uses causal mean pooling over sequences
    followed by MLP layers. Provides a fast, simple baseline without
    attention, RNN, or CNN complexity.

    Model ID: mlp4rec
    Model Type: Sequential

    Key Features:
        - Causal mean pooling (no future information leakage)
        - Configurable MLP architecture
        - Fast training and inference
        - Multiple pooling strategies
        - Optional residual connections

    Args:
        config (MLP4RecConfig): Model configuration with MLP parameters

    Example:
        >>> config = MLP4RecConfig(vocab_size=1000, hidden_dims=[256, 128])
        >>> model = MLP4Rec(config)
        >>> logits = model.forward(sequences, sequence_lengths)
    """

    def __init__(self, config: MLP4RecConfig):
        super().__init__(config)
        self.save_hyperparameters()
        
        self.ensemble_size = self.config.ensemble_size
        self.is_ensemble = self.ensemble_size > 1

        # Position embeddings
        self.pos_embedding = nn.Embedding(self.config.max_seq_length, self.embedding_dim)
        
        # Ensemble embedding adapters (if ensemble mode)
        if self.is_ensemble:
            self.embedding_r = nn.Parameter(torch.ones(self.ensemble_size, self.embedding_dim))
            self.embedding_s = nn.Parameter(torch.ones(self.ensemble_size, self.embedding_dim))

        # Learnable recency weights for mean pooling
        self.recency_weights = nn.Parameter(torch.zeros(self.config.max_seq_length))

        # Pre-MLP layer norm (adjust for multi-scale pooling: 2x for last+mean)
        mlp_input_dim = self.embedding_dim * 2 if self.config.pooling == "multi" else self.embedding_dim
        self.pre_mlp_norm = nn.LayerNorm(mlp_input_dim)

        # Build MLP layers (skip if empty for EmbeddingOnly)
        self.mlp_layers = nn.ModuleList()
        prev_dim = mlp_input_dim
        self.has_mlp = len(self.config.hidden_dims) > 0

        for hidden_dim in self.config.hidden_dims:
            # Use batch ensemble layers if ensemble mode
            if self.is_ensemble:
                self.mlp_layers.append(
                    LinearBatchEnsembleLayer(
                        in_features=prev_dim,
                        out_features=hidden_dim,
                        ensemble_size=self.ensemble_size,
                        ensemble_scaling_in=True,
                        ensemble_scaling_out=True,
                        ensemble_bias=False,
                        scaling_init="ones",
                    )
                )
            else:
                self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            if self.config.use_batch_norm:
                self.mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.config.use_layer_norm:
                self.mlp_layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            self.mlp_layers.append(self._get_activation())

            # Dropout
            self.mlp_layers.append(nn.Dropout(self.config.dropout_rate))

            prev_dim = hidden_dim

        # Final projection back to embedding dimension (only if MLP exists)
        if self.has_mlp:
            if self.is_ensemble:
                self.output_projection = LinearBatchEnsembleLayer(
                    in_features=prev_dim,
                    out_features=self.embedding_dim,
                    ensemble_size=self.ensemble_size,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=False,
                    scaling_init="ones",
                )
            else:
                self.output_projection = nn.Linear(prev_dim, self.embedding_dim)
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Store the actual embedding dim for predictions
        self.output_dim = self.embedding_dim

        # Attention pooling (if using attention pooling strategy)
        if self.config.pooling == "attention":
            self.attention_weights = nn.Linear(self.embedding_dim, 1)

        self._init_weights()

    def _get_activation(self):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(self.config.activation, nn.ReLU())

    def _init_weights(self):
        """Initialize model weights with better initialization."""
        nn.init.normal_(self.item_embedding.weight, std=self.config.init_std)
        nn.init.normal_(self.pos_embedding.weight, std=self.config.init_std)
        # Initialize recency weights with slight bias toward recent items
        nn.init.constant_(self.recency_weights, 0.0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization for better gradient flow
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sequences, sequence_lengths):
        """Forward pass - returns logits for ALL positions.
        
        Returns:
            Non-ensemble: [B, S, V]
            Ensemble: [B, S, E, V]
        """
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        
        if self.is_ensemble:
            # hidden_states: [B, S, E, D], compute logits per ensemble member
            logits = torch.einsum('bsed,vd->bsev', hidden_states, self.item_embedding.weight)
        else:
            logits = torch.matmul(hidden_states, self.item_embedding.weight.transpose(0, 1))
        return logits

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities.
        
        Returns:
            Non-ensemble: [B, V]
            Ensemble: [B, V] (averaged over ensemble members)
        """
        logits = self.forward(sequences, sequence_lengths)

        # Extract last valid position
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        last_logits = logits[batch_indices, last_indices]  # [B, V] or [B, E, V]

        if self.is_ensemble:
            # Average ensemble predictions: softmax per member, then average
            ensemble_probs = torch.softmax(last_logits, dim=-1)  # [B, E, V]
            return ensemble_probs.mean(dim=1)  # [B, V]
        else:
            return torch.softmax(last_logits, dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings."""
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)

        # Extract last valid position
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)

        return hidden_states[batch_indices, last_indices]

    def get_hidden_states(self, sequences, sequence_lengths):
        """Get hidden states with causal pooling + MLP.
        
        Returns:
            Non-ensemble: [B, S, D]
            Ensemble: [B, S, E, D]
        """
        batch_size, seq_len = sequences.size()

        # Get item embeddings: [B, S, D] or [B, S, E, D]
        embs = self.get_item_embedding(sequences)

        # Add position embeddings
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        positions = torch.clamp(positions, 0, self.config.max_seq_length - 1)
        pos_embs = self.pos_embedding(positions)  # [B, S, D]
        
        if self.is_ensemble:
            # Expand pos_embs to match ensemble dimension
            pos_embs = pos_embs.unsqueeze(2)  # [B, S, 1, D]
        embs = embs + pos_embs

        # Apply causal pooling
        pooled = self._causal_pool(embs, sequences)

        # If no MLP layers, return pooled embeddings directly
        if not self.has_mlp:
            return pooled

        # Pre-MLP normalization
        pooled = self.pre_mlp_norm(pooled)

        # Process through MLP (batch ensemble layers handle ensemble dimension automatically)
        x = pooled
        for i, layer in enumerate(self.mlp_layers):
            if isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)):
                x = layer(x)
            else:
                x = layer(x)

        # Final projection
        x = self.output_projection(x)

        # Residual connection (skip for ensemble due to dimension mismatch)
        if self.config.use_residual and not self.is_ensemble:
            if pooled.shape[-1] == x.shape[-1]:
                x = x + pooled

        x = self.final_layer_norm(x)

        return x

    def _causal_pool(self, embs, sequences):
        """Apply causal pooling to embeddings.

        Args:
            embs: [B, S, D] or [B, S, E, D] for ensemble
            sequences: [B, S] (for masking)

        Returns:
            [B, S, D] or [B, S, E, D] causally pooled embeddings
        """
        if self.config.pooling == "last":
            # Just return embeddings (no pooling)
            return embs

        elif self.config.pooling == "mean":
            # Causal mean pooling with learnable recency weights
            if self.is_ensemble:
                batch_size, seq_len, ensemble_size, dim = embs.shape
            else:
                batch_size, seq_len, dim = embs.shape
            mask = (sequences != 0).float()  # [B, S]
            
            # Apply learnable recency weights (softmax for normalization)
            recency = torch.softmax(self.recency_weights[:seq_len], dim=0)  # [S]
            
            # Weighted embeddings
            if self.is_ensemble:
                weighted_embs = embs * recency.view(1, -1, 1, 1)  # [B, S, E, D]
                masked_embs = weighted_embs * mask.unsqueeze(-1).unsqueeze(-1)  # [B, S, E, D]
            else:
                weighted_embs = embs * recency.view(1, -1, 1)  # [B, S, D]
                masked_embs = weighted_embs * mask.unsqueeze(-1)  # [B, S, D]

            # Cumulative sum for causal pooling
            cumsum_embs = torch.cumsum(masked_embs, dim=1)
            cumsum_weights = torch.cumsum(recency.unsqueeze(0) * mask, dim=1)  # [B, S]

            # Causal weighted mean
            if self.is_ensemble:
                pooled = cumsum_embs / (cumsum_weights.unsqueeze(-1).unsqueeze(-1) + 1e-9)
            else:
                pooled = cumsum_embs / (cumsum_weights.unsqueeze(-1) + 1e-9)

            return pooled

        elif self.config.pooling == "max":
            # Causal max pooling
            batch_size, seq_len, dim = embs.shape
            pooled = torch.zeros_like(embs)

            for i in range(seq_len):
                # Max over positions 0..i
                prefix = embs[:, :i+1, :]  # [B, i+1, D]
                mask = (sequences[:, :i+1] != 0).unsqueeze(-1)  # [B, i+1, 1]
                masked_prefix = prefix.masked_fill(~mask, float('-inf'))
                pooled[:, i, :] = masked_prefix.max(dim=1)[0]

            return pooled

        elif self.config.pooling == "attention":
            # Causal attention pooling
            batch_size, seq_len, dim = embs.shape

            # Compute attention scores
            attn_scores = self.attention_weights(embs).squeeze(-1)  # [B, S]

            # Apply causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=embs.device))
            causal_mask = causal_mask.unsqueeze(0)  # [1, S, S]

            # Mask padding
            padding_mask = (sequences != 0).float()  # [B, S]

            pooled = torch.zeros_like(embs)
            for i in range(seq_len):
                # Attention over positions 0..i
                scores = attn_scores[:, :i+1]  # [B, i+1]
                scores = scores.masked_fill(padding_mask[:, :i+1] == 0, float('-inf'))
                weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, i+1, 1]

                # Weighted sum
                pooled[:, i, :] = (embs[:, :i+1, :] * weights).sum(dim=1)

            return pooled

        elif self.config.pooling == "multi":
            # Multi-scale pooling: combine last + weighted mean (fast operations only)
            if self.is_ensemble:
                batch_size, seq_len, ensemble_size, dim = embs.shape
            else:
                batch_size, seq_len, dim = embs.shape
            
            # Last item (recency)
            last_pool = embs
            
            # Weighted mean pooling with learnable recency
            mask = (sequences != 0).float()
            recency = torch.softmax(self.recency_weights[:seq_len], dim=0)  # [S]
            
            if self.is_ensemble:
                weighted_embs = embs * recency.view(1, -1, 1, 1)  # [B, S, E, D]
                masked_embs = weighted_embs * mask.unsqueeze(-1).unsqueeze(-1)
                cumsum_embs = torch.cumsum(masked_embs, dim=1)
                cumsum_weights = torch.cumsum(recency.unsqueeze(0) * mask, dim=1)
                mean_pool = cumsum_embs / (cumsum_weights.unsqueeze(-1).unsqueeze(-1) + 1e-9)
            else:
                weighted_embs = embs * recency.view(1, -1, 1)  # [B, S, D]
                masked_embs = weighted_embs * mask.unsqueeze(-1)
                cumsum_embs = torch.cumsum(masked_embs, dim=1)
                cumsum_weights = torch.cumsum(recency.unsqueeze(0) * mask, dim=1)
                mean_pool = cumsum_embs / (cumsum_weights.unsqueeze(-1) + 1e-9)
            
            # Concatenate last + weighted mean: [B, S, 2*D] or [B, S, E, 2*D]
            return torch.cat([last_pool, mean_pool], dim=-1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific items.
        
        Returns:
            Non-ensemble: [B, S, D]
            Ensemble: [B, S, E, D]
        """
        clamped_ids = torch.clamp(item_ids, 0, self.item_embedding.num_embeddings - 1)
        base_emb = self.item_embedding(clamped_ids)  # [B, S, D]
        
        if self.is_ensemble:
            # Expand to ensemble dimension: [B, S, D] -> [B, S, E, D]
            B, S, D = base_emb.shape
            base_emb = base_emb.unsqueeze(2).expand(B, S, self.ensemble_size, D)
            # Apply per-member scaling: emb_k = s_k * base_emb * r_k
            ensemble_emb = self.embedding_s * base_emb * self.embedding_r
            return ensemble_emb  # [B, S, E, D]
        else:
            return base_emb  # [B, S, D]

    def compute_loss(self, batch):
        """Compute loss - sum over ensemble members if ensemble mode."""
        if not self.is_ensemble:
            # Use parent class implementation for non-ensemble
            return super().compute_loss(batch)
        
        # Ensemble mode: compute loss per member and sum
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)  # [B, S, E, D]
        
        targets, mask = self.get_targets_and_mask(batch)
        shifted_targets = targets[:, 1:]
        shifted_mask = mask[:, 1:]
        
        total_loss = 0
        for k in range(self.ensemble_size):
            # Extract k-th ensemble member
            ensemble_k_states = hidden_states[:, :-1, k, :]  # [B, S-1, D]
            
            # Compute logits
            ensemble_k_logits = torch.matmul(
                ensemble_k_states, self.item_embedding.weight.transpose(0, 1)
            )
            
            # Compute loss (same loss function for all members)
            loss_k = self.loss_fn(
                logits=ensemble_k_logits,
                targets=shifted_targets,
                mask=shifted_mask,
                neg_items=batch.get("neg_items"),
            )
            total_loss += loss_k
        
        return total_loss

    def get_targets_and_mask(self, batch):
        """Get targets and loss mask for causal prediction."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        targets = sequences
        batch_size, seq_len = sequences.size()
        mask = torch.zeros(batch_size, seq_len, device=sequences.device, dtype=torch.bool)

        for i, length in enumerate(sequence_lengths):
            if length > 1:
                mask[i, 1:length] = True

        return targets, mask
