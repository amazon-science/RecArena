import torch
from torch import nn


class SampledSoftmaxLoss(nn.Module):
    """Pure Sampled Softmax loss function.

    A stateless computational function that computes softmax over
    positive + sampled negative items (more efficient than full softmax).

    Expects pre-shifted and pre-processed inputs:
    - logits: [batch, seq_len, vocab_size] (pre-shifted, pre-processed) OR
    - hidden_states + item_embeddings (for fast sampled logits)
    - targets: [batch, seq_len] (pre-shifted, 0-indexed)
    - mask: [batch, seq_len] (pre-shifted)
    - neg_items: [batch, seq_len, num_neg] or [batch, num_neg] (required)
    """

    def __init__(self, ignore_index: int = -1, temperature: float = 1.0, l2_norm: bool = False):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )
        self.temperature = temperature
        self.l2_norm = l2_norm

    def __call__(self, logits=None, targets=None, mask=None, neg_items=None,
                 hidden_states=None, item_embeddings=None):
        """Compute sampled softmax loss.

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
                "Sampled Softmax requires negative samples in neg_items parameter"
            )

        batch_size, seq_len = targets.shape if targets.dim() == 2 else (targets.shape[0], 1)

        # Fast path: compute sampled logits directly from hidden states
        if hidden_states is not None and item_embeddings is not None:
            # Apply L2 normalization if enabled
            if self.l2_norm:
                hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
                item_embeddings = torch.nn.functional.normalize(item_embeddings, p=2, dim=-1)
            
            # Gather embeddings for positive and negative items
            pos_emb = item_embeddings[targets]  # [batch, seq_len, dim]
            
            # Compute positive scores
            pos_scores = (hidden_states * pos_emb).sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]
            
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
            
            # Apply temperature scaling
            pos_scores = pos_scores / self.temperature
            neg_scores = neg_scores / self.temperature
            
            # Combine: [batch, seq_len, 1 + num_neg]
            combined_scores = torch.cat([pos_scores, neg_scores], dim=-1)
        
        # Slow path: use pre-computed logits (backward compatible)
        elif logits is not None:
            # Gather positive scores
            pos_scores = torch.gather(
                logits, dim=2, index=targets.unsqueeze(-1)
            )  # [batch, seq_len, 1]

            # Gather negative scores
            if neg_items.dim() == 3:
                # Per-position negatives: [batch, seq_len, num_neg]
                neg_scores = torch.gather(
                    logits.unsqueeze(-2).expand(-1, -1, neg_items.size(-1), -1),
                    dim=3,
                    index=neg_items.unsqueeze(-1),
                ).squeeze(
                    -1
                )  # [batch, seq_len, num_neg]
            elif neg_items.dim() == 2:
                # Global negatives: [batch, num_neg]
                neg_items_expanded = neg_items.unsqueeze(1).expand(-1, seq_len, -1)
                neg_scores = torch.gather(
                    logits, dim=2, index=neg_items_expanded
                )  # [batch, seq_len, num_neg]
            else:
                raise ValueError(f"Unexpected neg_items shape: {neg_items.shape}")

            # Combine: [batch, seq_len, 1 + num_neg]
            combined_scores = torch.cat([pos_scores, neg_scores], dim=-1)
        else:
            raise ValueError(
                "Must provide either (hidden_states + item_embeddings) or logits"
            )

        # Target is always index 0 (the positive item)
        sampled_targets = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=combined_scores.device
        )

        # Compute cross-entropy over sampled items
        loss = self.cross_entropy(
            combined_scores.view(-1, combined_scores.size(-1)),
            sampled_targets.view(-1),
        )
        loss = loss.view(batch_size, seq_len) * mask

        return loss.sum() / mask.sum().clamp(min=1)
