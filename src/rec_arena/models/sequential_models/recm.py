import torch
import torch.nn as nn
from ...modules.layer_utils.batch_ensembling import LinearBatchEnsembleLayer
from ...modules.transformer_layers.transformer_block import TransformerBlock
from ..sequential import DeepSequentialModel
from ...configs.defaults.recm import RecMConfig


class RecM(DeepSequentialModel):

    # Ensemble compute_loss forms full-vocab logits (dense table gradient).
    SUPPORTS_SPARSE = False

    def __init__(self, config: RecMConfig):
        super().__init__(config)

        self.save_hyperparameters()

        # Learnable position embedding
        position_config = getattr(self.config, "position_config", {"type": "learnable"})
        if position_config["type"] == "rope":
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(
                self.config.max_seq_length, self.embedding_dim
            )
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

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
                    use_swiglu=self.config.use_ligr,  # Use SwiGLU when LiGR mode enabled
                    use_gated_residual=self.config.use_ligr,  # Use gated residuals when LiGR mode enabled
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5)
        self.dropout = nn.Dropout(self.config.dropout_rate)

        self.sequence_projection = nn.Sequential(
            LinearBatchEnsembleLayer(
                in_features=self.embedding_dim,
                out_features=self.embedding_dim * 4,
                ensemble_size=self.config.ensemble_size,
                ensemble_scaling_in=True,
                ensemble_scaling_out=True,
                ensemble_bias=False,
                scaling_init=self.config.scaling_init,
            ),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            LinearBatchEnsembleLayer(
                in_features=self.embedding_dim * 4,
                out_features=self.embedding_dim,
                ensemble_size=self.config.ensemble_size,
                ensemble_scaling_in=True,
                ensemble_scaling_out=True,
                ensemble_bias=False,
                scaling_init=self.config.scaling_init,
            ),
        )

        # Output embeddings (untied scoring table). When reuse_item_embeddings
        # is False, score against a SEPARATE table instead of the tied input
        # item_embedding. The table must have the SAME row count as the input
        # table (self.vocab_size, which already includes the special tokens) so
        # logits align with target item indices. self.num_items was never set on
        # this model (the base tracks self.vocab_size), so the previous
        # `self.num_items + 2` reference would crash the moment this branch ran.
        self.reuse_item_embeddings = self.config.reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(
                self.vocab_size, self.embedding_dim, padding_idx=0
            )

        # Pre-create loss functions for each ensemble member (avoid recreation overhead)
        if (
            hasattr(self.config, "ensemble_loss_functions")
            and self.config.ensemble_loss_functions
        ):
            from ...losses.factory import get_loss_function

            loss_functions = self.config.ensemble_loss_functions

            # Create one loss function per ensemble member
            self.ensemble_loss_fns = nn.ModuleList()
            for k in range(self.config.ensemble_size):
                if len(loss_functions) == 1:
                    loss_type = loss_functions[0]
                elif len(loss_functions) == self.config.ensemble_size:
                    loss_type = loss_functions[k]
                else:
                    members_per_loss = self.config.ensemble_size // len(loss_functions)
                    loss_idx = k // members_per_loss
                    loss_type = loss_functions[loss_idx]

                self.ensemble_loss_fns.append(get_loss_function(loss_type))

            # For Lightning display
            self.loss_fn = self.ensemble_loss_fns

        self._init_weights()

    def get_output_embeddings(self) -> torch.Tensor:
        """Output scoring table.

        Returns the untied output_embedding table when reuse_item_embeddings is
        False, otherwise the tied input item_embedding (base behavior). The
        default (reuse=True) path is unchanged.
        """
        if not self.reuse_item_embeddings:
            return self.output_embedding.weight
        return self.item_embedding.weight

    def forward(self, sequences, sequence_lengths):
        """Forward pass through RecM - returns logits for ALL positions and ensembles.

        NOTE: This creates a large (N, S, K, V) tensor. Only use for inference!
        For training, use compute_loss which is more memory-efficient.
        """
        # Get hidden states (N, S, K, D)
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)

        # Compute logits for all positions, ensembles and items using einsum
        # WARNING: This can be memory-intensive for large vocab!
        # Use get_output_embeddings() so the untied table is respected (identity
        # for the default reuse_item_embeddings=True path).
        logits = torch.einsum(
            "nskd,vd->nskv", hidden_states, self.get_output_embeddings()
        )

        return logits  # (N, S, K, V)

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item probabilities for the last position - OPTIMIZED."""
        # Get hidden states for all positions (N, S, K, D)
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)

        # Extract ONLY last valid position (much more efficient!)
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=sequences.size(1) - 1
        )
        last_hidden = hidden_states[batch_indices, last_indices]  # (N, K, D)

        # Compute logits only for last position
        last_logits = torch.einsum(
            "nkd,vd->nkv", last_hidden, self.get_output_embeddings()
        )  # (N, K, V)

        # Ensemble prediction: softmax per member, then average
        ensemble_probs = torch.softmax(last_logits, dim=-1)  # (N, K, V)
        avg_probs = ensemble_probs.mean(dim=1)  # (N, V)

        return avg_probs

    def get_sequence_embedding(self, sequences, sequence_lengths):
        """Get sequence-level embeddings - return ensemble embeddings for last position."""
        hidden_states = self.get_hidden_states(
            sequences, sequence_lengths
        )  # (N, S, K, D)

        # Extract last valid position for each ensemble
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=hidden_states.size(1) - 1
        )

        return hidden_states[batch_indices, last_indices]  # (N, K, D)

    def get_hidden_states(self, sequences, sequence_lengths):
        """Get hidden states from transformer - core SASRec computation."""
        batch_size, seq_len = sequences.size()

        # Get item embeddings (no scaling - ensemble projection handles transformation)
        item_embs = self.get_item_embedding(sequences)

        # Position embeddings
        if self.pos_embedding is not None:
            positions = torch.clamp(
                torch.arange(seq_len, device=sequences.device)
                .unsqueeze(0)
                .repeat(batch_size, 1),
                0,
                self.config.max_seq_length - 1,
            )
            pos_embs = self.pos_embedding(positions)
            x = self.dropout(item_embs + pos_embs)
        else:
            x = self.dropout(item_embs)

        # Apply transformer blocks with causal attention
        for block in self.transformer_blocks:
            x = block(x, attn_mask=None)

        # Apply layer norm BEFORE ensemble projection (like SASRec)
        x = self.layer_norm(x)

        # Apply batch ensemble projection with residual connection
        ensemble_embeds = self.sequence_projection(x)  # (N, S, K, D)

        # Add residual: broadcast x from (N, S, D) to (N, S, K, D)
        ensemble_embeds = ensemble_embeds + x.unsqueeze(2)

        return ensemble_embeds

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
            if length > 1:
                mask[i, 1:length] = True  # Predict positions 1 to length-1

        return targets, mask

    def get_loss_mask(self, batch):
        """Get loss mask for negative sampling losses."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        batch_size, seq_len = sequences.size()
        mask = torch.zeros(batch_size, seq_len, device=sequences.device)

        for i, length in enumerate(sequence_lengths):
            if length > 1:
                mask[i, 1:length] = 1.0  # Predict positions 1 to length-1

        return mask

    def compute_loss(self, batch):
        """Compute ensemble loss - different losses for different ensemble members.

        RecM applies the new loss signature per ensemble member.
        Each ensemble member k gets:
        - Its own logits from hidden_states[:, :, k, :]
        - Shared shifted targets (all predict same next items)
        - Its own negative samples (from neg_items_k)
        """
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        # MEMORY-EFFICIENT: Get hidden states once, compute logits per-member
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)

        # Get targets and mask (shared across ensemble)
        targets, mask = self.get_targets_and_mask(batch)

        # Apply causal shift
        shifted_targets = targets[:, 1:]  # Positions 1 to T
        shifted_mask = mask[:, 1:]  # Positions 1 to T

        total_loss = 0
        ensemble_size = hidden_states.size(2)  # K dimension

        # Use different loss functions for different ensemble members
        if (
            hasattr(self.config, "ensemble_loss_functions")
            and self.config.ensemble_loss_functions
        ):
            loss_functions = self.config.ensemble_loss_functions

            # Print ensemble loss configuration (only once during first call)
            if not hasattr(self, "_printed_ensemble_config"):
                print(f"\n🔧 RecM Ensemble Configuration:")
                print(f"   Ensemble Size: {ensemble_size}")
                print(f"   Loss Functions: {loss_functions}")
                members_per_loss = ensemble_size // len(loss_functions)
                for i, loss_fn in enumerate(loss_functions):
                    start_idx = i * members_per_loss
                    end_idx = start_idx + members_per_loss - 1
                    print(f"   Ensemble heads {start_idx}-{end_idx}: {loss_fn.upper()}")
                if len(set(loss_functions)) > 1:
                    print(
                        f"   ⚖️  Adaptive loss scaling enabled (different loss functions detected)"
                    )
                print()
                self._printed_ensemble_config = True

            # Check if we need loss scaling (only if using different loss functions)
            use_loss_scaling = len(set(loss_functions)) > 1

            # Compute losses for each ensemble member
            losses = []
            for k in range(ensemble_size):
                # Extract k-th ensemble member's hidden states: (N, S-1, D)
                ensemble_k_states = hidden_states[:, :-1, k, :]  # Apply shift here

                # Get negative samples for this ensemble member
                neg_key = f"neg_items_{k}"
                ensemble_neg_items = batch.get(neg_key)

                # Shift negatives if per-position
                if ensemble_neg_items is not None:
                    if ensemble_neg_items.dim() == 3:
                        ensemble_neg_items = ensemble_neg_items[:, 1:, :]
                    elif (
                        ensemble_neg_items.dim() == 2 and ensemble_neg_items.size(1) > 1
                    ):
                        if ensemble_neg_items.size(1) == targets.size(1):
                            ensemble_neg_items = ensemble_neg_items[:, 1:]

                # Use pre-created loss function for this ensemble member
                ensemble_loss_fn = self.ensemble_loss_fns[k]
                loss_type = (
                    loss_functions[k]
                    if len(loss_functions) == ensemble_size
                    else (
                        loss_functions[k // (ensemble_size // len(loss_functions))]
                        if len(loss_functions) > 1
                        else loss_functions[0]
                    )
                )

                # OPTIMIZATION: Only compute full logits for cross_entropy
                # Other losses use hidden_states + item_embeddings directly
                if loss_type == "cross_entropy":
                    ensemble_k_logits = torch.matmul(
                        ensemble_k_states, self.get_output_embeddings().transpose(0, 1)
                    )
                    loss_k = ensemble_loss_fn(
                        logits=ensemble_k_logits,
                        targets=shifted_targets,
                        mask=shifted_mask,
                        neg_items=ensemble_neg_items,
                    )
                else:
                    # Pass hidden_states for memory-efficient computation
                    loss_k = ensemble_loss_fn(
                        hidden_states=ensemble_k_states,
                        item_embeddings=self.get_output_embeddings(),
                        targets=shifted_targets,
                        mask=shifted_mask,
                        neg_items=ensemble_neg_items,
                    )
                losses.append((loss_k, loss_type))

            # Second pass: apply adaptive scaling if needed
            if use_loss_scaling:
                # Compute mean loss magnitude per loss type (detached for efficiency)
                loss_type_magnitudes = {}
                for loss_val, loss_type in losses:
                    mag = loss_val.detach()
                    # Skip NaN losses
                    if torch.isnan(mag) or torch.isinf(mag):
                        continue
                    if loss_type not in loss_type_magnitudes:
                        loss_type_magnitudes[loss_type] = mag
                    else:
                        loss_type_magnitudes[loss_type] = (
                            loss_type_magnitudes[loss_type] + mag
                        )

                # Target: normalize all losses to have similar magnitude
                if loss_type_magnitudes:
                    target_magnitude = sum(loss_type_magnitudes.values()) / len(
                        loss_type_magnitudes
                    )
                else:
                    target_magnitude = 1.0  # Fallback if all losses are NaN

                # Apply scaling in single pass
                for loss_val, loss_type in losses:
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        continue  # Skip NaN/Inf losses
                    mag = loss_type_magnitudes.get(loss_type, 1.0)
                    scale = target_magnitude / (mag + 1e-8)
                    total_loss = total_loss + scale * loss_val
            else:
                # No scaling needed, just sum
                for loss_val, _ in losses:
                    if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                        total_loss = total_loss + loss_val
        else:
            # Default: use same loss for all ensemble members
            for k in range(ensemble_size):
                # Extract and shift k-th ensemble member
                ensemble_k_states = hidden_states[:, :-1, k, :]

                # Compute logits (memory-efficient!)
                ensemble_k_logits = torch.matmul(
                    ensemble_k_states, self.get_output_embeddings().transpose(0, 1)
                )

                # Get negatives if present
                neg_items = batch.get("neg_items")
                if neg_items is not None and neg_items.dim() >= 2:
                    # Shift if per-position
                    if neg_items.dim() == 3:
                        neg_items = neg_items[:, 1:, :]
                    elif neg_items.size(1) == targets.size(1):
                        neg_items = neg_items[:, 1:]

                # Compute loss with new signature
                loss_k = self.loss_fn(
                    logits=ensemble_k_logits,
                    targets=shifted_targets,
                    mask=shifted_mask,
                    neg_items=neg_items,
                )
                total_loss += loss_k

        # Average loss across ensemble members (not sum)
        return total_loss / ensemble_size

    def validation_step(self, batch, batch_idx):
        """Override validation_step to handle ensemble losses properly."""
        if "target" in batch:
            sequences = batch["sequence"]
            sequence_lengths = batch["sequence_length"]
            targets = batch["target"]

            # Get predictions (averaged over ensemble)
            predictions = self.predict_next(sequences, sequence_lengths)

            # Compute accuracy@10
            _, top10 = torch.topk(predictions, k=10, dim=-1)
            acc10 = (top10 == targets.unsqueeze(1)).any(dim=1).float().mean()
            self.log("val_acc@10", acc10, on_step=False, on_epoch=True, prog_bar=True)

            return acc10
        else:
            # Fallback: compute loss using compute_loss
            loss = self.compute_loss(batch)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def test_step(self, batch, batch_idx):
        """Override test_step to properly handle ensemble predictions."""
        if "target" in batch:
            sequences = batch["sequence"]
            sequence_lengths = batch["sequence_length"]
            targets = batch["target"]

            # Get predictions (averaged over ensemble)
            predictions = self.predict_next(sequences, sequence_lengths)

            # Compute accuracy@10
            _, top10 = torch.topk(predictions, k=10, dim=-1)
            acc10 = (top10 == targets.unsqueeze(1)).any(dim=1).float().mean()
            self.log("test_acc@10", acc10, on_step=False, on_epoch=True, prog_bar=True)

            # Compute additional metrics
            if self._metric_calculator is None:
                from ...metrics import MetricCalculator

                self._metric_calculator = MetricCalculator(k_values=self.val_k_values)

            predictions_cpu = predictions.detach().cpu()
            targets_cpu = targets.detach().cpu()
            metrics = self._metric_calculator.calculate_all(
                predictions_cpu, targets_cpu
            )

            for metric_name, value in metrics.items():
                self.log(
                    f"test_{metric_name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

            # Return predictions and targets for consistency
            return {
                "predictions": predictions_cpu,
                "targets": targets_cpu,
                "test_acc@10": acc10,
            }
        else:
            # Fallback to base implementation
            return super().test_step(batch, batch_idx)

    def _init_weights(self):
        """Initialize weights properly like SASRec."""
        # Initialize embeddings
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # Untied output scoring table (only present when reuse_item_embeddings
        # is False) gets the same small-std init + zeroed PAD row.
        if not self.reuse_item_embeddings:
            nn.init.normal_(self.output_embedding.weight, std=0.02)
            with torch.no_grad():
                self.output_embedding.weight[0].zero_()

        # Note: LinearBatchEnsembleLayer handles its own initialization in reset_parameters()
        # We only need to initialize standard nn.Linear layers here
        for module in self.sequence_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
