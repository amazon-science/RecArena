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
        position_config = getattr(config, "position_config", {"type": "learnable"})
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

        norm_eps = getattr(self.config, "layer_norm_eps", 1e-5)
        use_bias = getattr(self.config, "use_bias", False)

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
                    norm_first=getattr(self.config, "layer_norm_first", True),
                    bias=use_bias,
                    norm_eps=norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Optional input-embedding LayerNorm (RecBole-faithful; default off).
        self.input_layer_norm = (
            nn.LayerNorm(self.embedding_dim, eps=norm_eps)
            if getattr(self.config, "input_layer_norm", False)
            else None
        )
        # Final layer norm after blocks; suppressible for RecBole-faithful mode.
        self.layer_norm = (
            nn.LayerNorm(self.embedding_dim, eps=norm_eps)
            if getattr(self.config, "final_layer_norm", True)
            else None
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        # Output prediction head (Linear -> GELU -> LayerNorm) applied to the
        # encoder output before tied scoring, matching original BERT4Rec/RecBole.
        if getattr(self.config, "output_head", False):
            self.out_ffn = nn.Linear(self.embedding_dim, self.embedding_dim)
            self.out_gelu = nn.GELU()
            self.out_ln = nn.LayerNorm(self.embedding_dim, eps=norm_eps)
        else:
            self.out_ffn = None
        # Learned per-item output bias (popularity prior) added to every logit.
        self.out_bias = (
            nn.Parameter(torch.zeros(self.vocab_size))
            if getattr(self.config, "output_bias", False)
            else None
        )

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
        logits = torch.matmul(
            hidden_states, self.get_output_embeddings().transpose(0, 1)
        )
        # Learned per-item bias (popularity prior), matching RecBole BERT4Rec.
        if self.out_bias is not None:
            logits = logits + self.out_bias

        return logits

    def predict_next(self, sequences, sequence_lengths):
        """Predict the next item, BERT4Rec-style.

        Per the paper, inference appends a ``[MASK]`` token to the END of the
        sequence (position = sequence_length) and predicts that slot from its
        final hidden state. This is different from masking the last *real* item
        (which would reconstruct an already-seen interaction, not the next one).

        When a sequence already fills ``max_seq_length``, we drop its oldest item
        to make room for the trailing ``[MASK]`` (standard sliding-window eval).
        """
        batch_size, seq_len = sequences.size()
        device = sequences.device
        masked_sequences = sequences.clone()
        batch_indices = torch.arange(batch_size, device=device)

        # Append the MASK at index = sequence_length (one PAST the last real
        # item), keeping ALL real items as context, so the MASK slot predicts the
        # NEXT (held-out test) item -- the correct LOO next-item task. (We must
        # NOT mask a real item at index length-1: that would destroy the last
        # input item and ask the model to reconstruct it, but the eval target is
        # the next item, not that one.) On overflow (seq already full), drop the
        # oldest item to make room, mirroring RecBole's reconstruct_test_data.
        mask_pos = sequence_lengths.clone()
        overflow = mask_pos >= seq_len
        if overflow.any():
            masked_sequences[overflow] = torch.roll(
                masked_sequences[overflow], shifts=-1, dims=1
            )
            mask_pos[overflow] = seq_len - 1
        mask_pos = torch.clamp(mask_pos, 0, seq_len - 1)
        masked_sequences[batch_indices, mask_pos] = self.mask_token_id

        # MASK is a valid token; attend over all non-padding positions.
        attention_mask = masked_sequences != 0
        logits = self.forward(masked_sequences, attention_mask=attention_mask)

        # Predictions at the appended MASK position.
        last_logits = logits[batch_indices, mask_pos]  # [batch_size, vocab_size]

        # Remove special tokens from predictions.
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
        """Apply BERT-style masking to sequences for external loss functions.

        In addition to the random Cloze masking, for a fraction of sequences we
        force-mask the LAST real item (``last_item_mask_prob``). Training
        otherwise only masks interior positions (which have both left and right
        context), while inference always predicts a trailing ``[MASK]`` with
        left context only -- a train/eval mismatch that hurts next-item ranking.
        This auxiliary task (from the BERT4Rec paper) closes that gap.
        """
        if mask_prob is None:
            mask_prob = self.config.mask_prob

        masked_sequences = sequences.clone()
        labels = torch.full_like(sequences, -100)  # -100 is ignored in loss

        # RecBole-faithful masking: a BATCH-LEVEL mutually-exclusive switch (NOT
        # a superposition). With prob last_item_mask_prob (== RecBole ft_ratio),
        # the WHOLE batch does PURE last-item masking -- mask only each seq's
        # final real item, intact left context -- which is EXACTLY the eval task
        # (predict a trailing MASK from left context). Otherwise the whole batch
        # does PURE interior Cloze. The two never mix in one forward pass; mixing
        # them (the prior behavior) meant the model never cleanly trained the
        # eval task, since last-item-masked seqs also had interior items masked.
        # Mirrors RecBole transform.py:138-140 (batch coin flip) + _append_mask_last.
        last_prob = getattr(self.config, "last_item_mask_prob", 0.0)
        if last_prob > 0 and float(torch.rand(1).item()) < last_prob:
            lengths = (sequences != 0).sum(dim=1)  # real length per row
            last_idx = torch.clamp(lengths - 1, min=0)
            rows = torch.arange(sequences.size(0), device=sequences.device)
            has_items = lengths > 0
            r = rows[has_items]
            c = last_idx[has_items]
            labels[r, c] = torch.clamp(sequences[r, c], 0, self.vocab_size - 1)
            masked_sequences[r, c] = self.mask_token_id
            return masked_sequences, labels

        # --- else: PURE interior Cloze masking ---
        # Create random mask (don't mask padding tokens)
        mask_prob_matrix = torch.rand(sequences.shape, device=sequences.device)
        padding_mask = sequences == 0  # PAD_TOKEN = 0
        # Set padding positions to 1.0 so they're never selected (since we check < mask_prob)
        mask_prob_matrix.masked_fill_(padding_mask, 1.0)

        masked_indices = mask_prob_matrix < mask_prob

        # Mask FLOOR: guarantee >=1 masked position per non-empty sequence. With
        # pure Bernoulli(mask_prob=0.2) on short ml-100k sequences, some rows get
        # ZERO masked positions -> that example contributes no loss, weakening
        # the per-batch signal and adding large seed variance. For each non-empty
        # row with no mask, force one uniformly-random real (non-PAD) position.
        real = ~padding_mask
        no_mask = real.any(dim=1) & (~masked_indices.any(dim=1))
        if no_mask.any():
            # Pick a random real position per offending row via argmax on random
            # scores restricted to real positions.
            scores = torch.rand(sequences.shape, device=sequences.device)
            scores.masked_fill_(padding_mask, -1.0)
            pick = scores.argmax(dim=1)  # [batch] a real col per row
            rows = torch.arange(sequences.size(0), device=sequences.device)[no_mask]
            masked_indices[rows, pick[no_mask]] = True

        # Clamp labels to valid vocab range
        valid_tokens = torch.clamp(sequences[masked_indices], 0, self.vocab_size - 1)
        labels[masked_indices] = valid_tokens

        # mask_token_prob% of the time, replace with [MASK] token.
        # dtype=float is REQUIRED: labels.shape comes from a Long tensor and
        # torch.bernoulli needs a float probability tensor. A float fill value
        # (e.g. 0.8) happens to infer float, but an int fill value (e.g. the
        # random_prob=0 branch below when mask_token_prob==1.0) would infer Long
        # and crash bernoulli. Pin float explicitly on both.
        indices_replaced = (
            torch.bernoulli(
                torch.full(
                    labels.shape,
                    float(self.config.mask_token_prob),
                    device=sequences.device,
                    dtype=torch.float,
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
            else 0.0
        )
        indices_random = (
            torch.bernoulli(
                torch.full(
                    labels.shape,
                    float(random_prob),
                    device=sequences.device,
                    dtype=torch.float,
                )
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
        if getattr(self.config, "scale_embeddings", True):
            item_embs = item_embs * (self.embedding_dim**0.5)

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
        # RecBole-faithful: LayerNorm the input embeddings before dropout.
        if self.input_layer_norm is not None:
            x = self.input_layer_norm(x)
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

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        # Output prediction head (Linear -> GELU -> LayerNorm) before tied
        # scoring. Applied here so both forward() and compute_loss() (which both
        # call get_hidden_states) share the same scoring representation.
        if self.out_ffn is not None:
            x = self.out_ln(self.out_gelu(self.out_ffn(x)))
        return x

    def _init_weights(self):
        """Initialize ALL modules like RecBole BERT4Rec (self.apply(_init_weights),
        bert4rec.py:97-107): every nn.Linear and nn.Embedding weight ~ N(0, std),
        Linear/LayerNorm biases zeroed, LayerNorm weight = 1.

        Previously only the item/position embeddings were initialized; the whole
        transformer encoder (qkv/out projections, FFN linears), the output head
        (out_ffn/out_ln), and input_layer_norm were left at PyTorch defaults --
        nn.Linear defaults to kaiming_uniform_(a=sqrt(5)) (std ~1/sqrt(fan_in)
        ~= 0.125 here, ~6x RecBole's 0.02) with uniform-random biases. That
        degraded the from-scratch optimum and made results highly seed-sensitive.
        The weight-parity test could never catch this: it copies RecBole's
        TRAINED weights in, bypassing init entirely.
        """
        std = getattr(self.config, "init_std", 0.02)

        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(_init)
        # Keep the PAD row of the item embedding at zero (padding_idx semantics).
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

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

        # Non-masked positions are excluded two ways: (1) the `mask` zeroes
        # their per-token loss, and (2) we set their target to 0 (PAD) which the
        # CE loss treats as ignore_index. (Using -100 here would crash CE whose
        # ignore_index is 0, not -100.) Sampled losses rely on `mask` only.
        targets = torch.where(
            labels == -100,
            torch.zeros_like(labels),
            torch.clamp(labels, 0, self.vocab_size - 1),
        )

        # Extract negative samples if present
        neg_items = batch.get("neg_items")
        # Note: neg_items don't need shifting for BERT4Rec (in-place prediction)

        # Call loss function with hidden states for fast sampled logits.
        # embedding_lookup is passed only for sparse tables (sampled losses).
        extra = {}
        lookup = self.embedding_lookup()
        if lookup is not None:
            extra["embedding_lookup"] = lookup
        # Thread the learned per-item output bias into the TRAINING logits so it
        # is trained, not just applied at inference (forward()/predict_next()).
        # RecBole BERT4Rec adds this bias in BOTH calculate_loss and
        # full_sort_predict. Only the full-softmax CrossEntropyLoss path forms
        # logits here and accepts the output_bias kwarg; sampled losses form
        # their own (sampled) logits and are left unchanged.
        if self.out_bias is not None:
            from ...losses.sequential.cross_entropy import CrossEntropyLoss

            if isinstance(self.loss_fn, CrossEntropyLoss):
                extra["output_bias"] = self.out_bias
        return self.loss_fn(
            hidden_states=hidden_states,
            item_embeddings=self.get_output_embeddings(),
            targets=targets,
            mask=mask,
            neg_items=neg_items,
            **extra,
        )
