import torch
from torch import nn


class BPRLoss(nn.Module):
    """Pure BPR loss function.

    A stateless computational function that expects:
    - logits: [batch, seq_len, vocab_size] (pre-shifted, pre-processed) OR
    - hidden_states + item_embeddings (for fast sampled logits)
    - targets: [batch, seq_len] (pre-shifted, 0-indexed)
    - mask: [batch, seq_len] (pre-shifted)
    - neg_items: [batch, seq_len, num_neg] or [batch, num_neg] (required)
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logits=None, targets=None, mask=None, neg_items=None,
                 hidden_states=None, item_embeddings=None):
        """Compute BPR loss with negative sampling.

        Args:
            logits: [batch, seq_len, vocab_size] (optional, for backward compatibility)
            targets: [batch, seq_len] (0-indexed)
            mask: [batch, seq_len]
            neg_items: [batch, seq_len, num_neg] or [batch, num_neg] (required)
            hidden_states: [batch, seq_len, dim] (optional, for fast path)
            item_embeddings: [vocab_size, dim] (optional, for fast path)

        Returns:
            Scalar loss value
        """
        if neg_items is None:
            raise ValueError(
                "BPR loss requires negative samples in neg_items parameter"
            )

        batch_size, seq_len = targets.shape if targets.dim() == 2 else (targets.shape[0], 1)

        # Fast path: compute sampled logits directly from hidden states
        if hidden_states is not None and item_embeddings is not None:
            # Gather embeddings for positive and negative items
            pos_emb = item_embeddings[targets]  # [batch, seq_len, dim]
            
            # Compute positive scores
            pos_scores = (hidden_states * pos_emb).sum(dim=-1)  # [batch, seq_len]
            
            # Handle negative items (2D or 3D)
            if neg_items.dim() == 3:
                # Per-position negatives: [batch, seq_len, num_neg]
                neg_emb = item_embeddings[neg_items]  # [batch, seq_len, num_neg, dim]
                neg_scores = torch.einsum('bsd,bsnd->bsn', hidden_states, neg_emb)  # [batch, seq_len, num_neg]
            elif neg_items.dim() == 2:
                # Global negatives: [batch, num_neg]
                neg_emb = item_embeddings[neg_items]  # [batch, num_neg, dim]
                neg_scores = torch.einsum('bsd,bnd->bsn', hidden_states, neg_emb)  # [batch, seq_len, num_neg]
            else:
                raise ValueError(f"Unexpected neg_items shape: {neg_items.shape}")
        
        # Slow path: use pre-computed logits (backward compatible)
        elif logits is not None:
            vocab_size = logits.shape[-1]
            
            # Gather positive scores: logits[b, t, targets[b, t]]
            pos_scores = torch.gather(logits, dim=2, index=targets.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]

            # Extract negative scores from logits
            if neg_items.dim() == 3:
                # Per-position negatives: [batch, seq_len, num_neg]
                neg_scores = torch.gather(logits, dim=2, index=neg_items)  # [batch, seq_len, num_neg]
            elif neg_items.dim() == 2:
                # Global negatives per batch: [batch, num_neg]
                neg_items_expanded = neg_items.unsqueeze(1).expand(-1, seq_len, -1)
                neg_scores = torch.gather(logits, dim=2, index=neg_items_expanded)  # [batch, seq_len, num_neg]
            else:
                raise ValueError(f"Unexpected neg_items shape: {neg_items.shape}")
        else:
            raise ValueError(
                "Must provide either (hidden_states + item_embeddings) or logits"
            )

        # Apply BPR loss: -log(sigmoid(pos_score - neg_score))
        # Use logsigmoid for numerical stability
        pos_scores = pos_scores.unsqueeze(-1)  # [batch, seq_len, 1]
        diff = pos_scores - neg_scores  # [batch, seq_len, num_neg]
        loss = -torch.nn.functional.logsigmoid(diff)

        # Average over negatives, apply mask
        loss = loss.mean(dim=-1) * mask  # [batch, seq_len]

        return loss.sum() / mask.sum().clamp(min=1)
