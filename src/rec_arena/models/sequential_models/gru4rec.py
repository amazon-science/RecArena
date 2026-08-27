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

        # Optional Linear(hidden_size -> embedding_dim) before tied scoring
        # (standard GRU4Rec / RecBole design). Lets the GRU be wider than the
        # embeddings. When absent, hidden_size == embedding_dim is enforced.
        self.hidden_projection = (
            nn.Linear(self.config.hidden_size, self.embedding_dim)
            if getattr(self.config, "use_hidden_projection", False)
            else None
        )

        # RecBole-faithful dropout placement: emb_dropout on the INPUT item
        # embeddings (before the GRU) ONLY. RecBole's GRU4Rec applies dropout
        # solely on the input embeddings and has NO dropout between the GRU and
        # the dense projection (gru4rec.py:80). It lives inside get_hidden_states
        # so TRAINING (base compute_loss -> get_hidden_states) and EVAL (forward
        # -> get_hidden_states) are regularized identically. (An earlier fix in
        # this project also added dropout on the GRU OUTPUT, which double-
        # regularized vs RecBole at the same nominal rate and cost NDCG on small
        # data -- removed to match RecBole exactly.)
        self.emb_dropout = nn.Dropout(self.config.dropout_rate)

        # Weight init mirroring RecBole GRU4Rec (gru4rec.py:71-76): xavier_normal_
        # on the item embedding, xavier_uniform_ on the GRU input/hidden weight
        # matrices. Previously the GRU weights were left at PyTorch's default
        # U(-1/sqrt(h), 1/sqrt(h)) and the embedding used a fixed N(0,init_std),
        # both diverging from RecBole and hurting from-scratch convergence.
        # Guarded by apply_init to preserve old behavior if explicitly disabled.
        if getattr(self.config, "apply_init", True):
            nn.init.xavier_normal_(self.item_embedding.weight)
            with torch.no_grad():
                self.item_embedding.weight[0].zero_()  # keep PAD row at 0
            # Xavier-init every GRU weight matrix (weight_ih_l*, weight_hh_l*),
            # matching RecBole's xavier_uniform_ on the GRU parameters.
            for name, param in self.gru.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_uniform_(param)
                elif name.startswith("bias"):
                    nn.init.zeros_(param)
            if self.hidden_projection is not None:
                nn.init.xavier_uniform_(self.hidden_projection.weight)
                nn.init.zeros_(self.hidden_projection.bias)

    def forward(self, sequences, sequence_lengths):
        """Forward pass through GRU4Rec - returns logits for ALL positions."""
        # Get all hidden states: [batch_size, seq_len, hidden_size]
        # NOTE: emb_dropout is applied INSIDE get_hidden_states so it is shared
        # by both the eval (forward) and train (compute_loss) paths. Do NOT
        # re-apply dropout here.
        all_hidden = self.get_hidden_states(sequences, sequence_lengths)

        # Compute logits for ALL positions (needed for per-position loss)
        logits = torch.matmul(all_hidden, self.get_output_embeddings().transpose(0, 1))

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
        # RecBole-faithful: dropout on the INPUT embeddings before the GRU.
        embedded = self.emb_dropout(embedded)

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

        # NOTE: no dropout on the GRU output -- RecBole applies dropout only on
        # the input embeddings (see __init__). The output goes straight to the
        # (optional) projection / tied scoring.

        # Project GRU hidden state to embedding space for tied scoring (standard
        # GRU4Rec). No-op when the projection is disabled (hidden==emb).
        if self.hidden_projection is not None:
            output = self.hidden_projection(output)

        return output  # [batch_size, seq_len, embedding_dim]

    def compute_loss(self, batch):
        """Loss for GRU4Rec.

        Default (last_position_loss=False): defer to the base per-position
        causal-shift CE (predict every position i+1 -- the SASRec-style
        objective shared across RecArena sequential models).

        last_position_loss=True: the canonical GRU4Rec / RecBole objective --
        gather the final hidden state and predict the single next item with full
        cross-entropy. Each sequence contributes ONE supervised target (the
        held-out next item), matching RecBole's data-augmentation recipe where
        every prefix is its own example scored at its last step.
        """
        if not getattr(self.config, "last_position_loss", False):
            return super().compute_loss(batch)

        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        # Predict the item at the last valid position from the prefix before it.
        # The input to the GRU is the sequence up to length-1; the target is the
        # item at length-1 (the next item relative to that prefix).
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0)
        targets = sequences[batch_indices, last_indices]  # [batch]

        # Feed the prefix (drop the final item) so the last hidden state predicts
        # it, mirroring RecBole's gather at item_seq_len-1.
        prefix = sequences.clone()
        prefix[batch_indices, last_indices] = 0
        prefix_lengths = torch.clamp(sequence_lengths - 1, min=1)
        hidden = self.get_hidden_states(prefix, prefix_lengths)  # [B, S, emb]
        last_hidden = hidden[batch_indices, torch.clamp(prefix_lengths - 1, min=0)]
        logits = torch.matmul(last_hidden, self.get_output_embeddings().transpose(0, 1))

        # Full-vocab CE on the single target (ignore PAD target=0).
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=0)

    def get_targets_and_mask(self, batch):
        """Get targets and mask for causal prediction."""
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]

        # Targets are the sequences themselves (will be shifted in compute_loss)
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
