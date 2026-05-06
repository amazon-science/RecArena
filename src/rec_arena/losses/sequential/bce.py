import torch
from torch import nn


class BCENegativeSamplingLoss(nn.Module):
    """BCE loss with negative sampling for sequential recommendation.

    Combines positive + negative logits into [batch, seq_len, 1 + num_neg]
    and applies BCE with labels [1, 0, 0, ...] (same pattern as GBCE).
    """

    def __init__(self, l2_norm: bool = False):
        super().__init__()
        self.l2_norm = l2_norm

    def __call__(
        self,
        logits=None,
        targets=None,
        mask=None,
        neg_items=None,
        hidden_states=None,
        item_embeddings=None,
    ):
        if neg_items is None:
            raise ValueError("BCE loss requires neg_items")

        batch_size, seq_len = (
            targets.shape if targets.dim() == 2 else (targets.shape[0], 1)
        )

        # Fast path: compute from hidden states
        if hidden_states is not None and item_embeddings is not None:
            if self.l2_norm:
                hidden_states = nn.functional.normalize(hidden_states, p=2, dim=-1)
                item_embeddings = nn.functional.normalize(item_embeddings, p=2, dim=-1)

            pos_emb = item_embeddings[targets]  # [batch, seq_len, dim]
            pos_scores = (hidden_states * pos_emb).sum(
                dim=-1, keepdim=True
            )  # [batch, seq_len, 1]

            if neg_items.dim() == 3:
                neg_emb = item_embeddings[neg_items]  # [batch, seq_len, num_neg, dim]
                neg_scores = torch.einsum("bsd,bsnd->bsn", hidden_states, neg_emb)
            else:  # 2D: [batch, num_neg]
                neg_emb = item_embeddings[neg_items]  # [batch, num_neg, dim]
                neg_scores = torch.einsum("bsd,bnd->bsn", hidden_states, neg_emb)

            combined_logits = torch.cat([pos_scores, neg_scores], dim=-1)

        # Slow path: from pre-computed logits
        elif logits is not None:
            pos_scores = torch.gather(
                logits, dim=2, index=targets.unsqueeze(-1)
            )  # [batch, seq_len, 1]

            if neg_items.dim() == 3:
                neg_scores = torch.gather(logits, dim=2, index=neg_items)
            else:  # 2D
                neg_items_expanded = neg_items.unsqueeze(1).expand(-1, seq_len, -1)
                neg_scores = torch.gather(logits, dim=2, index=neg_items_expanded)

            combined_logits = torch.cat([pos_scores, neg_scores], dim=-1)
        else:
            raise ValueError("Must provide (hidden_states + item_embeddings) or logits")

        # Labels: [1, 0, 0, ...] for [pos, neg, neg, ...]
        labels = torch.zeros_like(combined_logits)
        labels[:, :, 0] = 1.0

        # BCE loss averaged over pos+neg, then masked
        loss = (
            nn.functional.binary_cross_entropy_with_logits(
                combined_logits, labels, reduction="none"
            ).mean(dim=-1)
            * mask
        )

        return loss.sum() / mask.sum().clamp(min=1)
