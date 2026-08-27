import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """Pure cross-entropy loss function.

    A stateless computational function that expects:
    - logits: [batch, seq_len, vocab_size] (pre-shifted, pre-processed) OR
    - hidden_states + item_embeddings (will compute full logits)
    - targets: [batch, seq_len] (pre-shifted, 0-indexed)
    - mask: [batch, seq_len] (pre-shifted)
    - neg_items: Not used (provided for API consistency)
    """

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        # Use 0 as ignore_index since padding is 0
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def __call__(self, logits=None, targets=None, mask=None, neg_items=None,
                 hidden_states=None, item_embeddings=None, output_bias=None):
        """Compute cross-entropy loss.

        Args:
            logits: [batch, seq_len, vocab_size] (optional, for backward compatibility)
            targets: [batch, seq_len] (0-indexed)
            mask: [batch, seq_len]
            neg_items: Not used (API consistency)
            hidden_states: [batch, seq_len, dim] (optional)
            item_embeddings: [vocab_size, dim] (optional)
            output_bias: [vocab_size] optional per-item additive logit bias
                (popularity prior). When provided and logits are computed from
                hidden_states here, it is added to every logit so the bias is
                TRAINED, matching how it is applied at inference (BERT4Rec /
                RecBole). Default None = no bias (unchanged behavior).

        Returns:
            Scalar loss value
        """
        # Compute full vocab logits if hidden_states provided
        if logits is None:
            if hidden_states is not None and item_embeddings is not None:
                logits = torch.matmul(hidden_states, item_embeddings.transpose(0, 1))
                if output_bias is not None:
                    logits = logits + output_bias
            else:
                raise ValueError(
                    "Must provide either logits or (hidden_states + item_embeddings)"
                )
        
        # Flatten and compute loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        loss = self.cross_entropy(logits_flat, targets_flat) * mask_flat.float()

        return loss.sum() / mask_flat.float().sum().clamp(min=1)
