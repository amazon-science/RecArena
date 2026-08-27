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
        embedding_lookup=None,
    ):
        if neg_items is None:
            raise ValueError("BCE loss requires neg_items")

        batch_size, seq_len = (
            targets.shape if targets.dim() == 2 else (targets.shape[0], 1)
        )

        # Fast path: compute from hidden states
        if hidden_states is not None and item_embeddings is not None:
            lookup = (
                embedding_lookup
                if embedding_lookup is not None
                else (lambda ids: item_embeddings[ids])
            )
            if self.l2_norm:
                hidden_states = nn.functional.normalize(hidden_states, p=2, dim=-1)

            pos_emb = lookup(targets)  # [batch, seq_len, dim]
            if self.l2_norm:
                pos_emb = nn.functional.normalize(pos_emb, p=2, dim=-1)
            pos_scores = (hidden_states * pos_emb).sum(
                dim=-1, keepdim=True
            )  # [batch, seq_len, 1]

            neg_emb = lookup(neg_items)
            if self.l2_norm:
                neg_emb = nn.functional.normalize(neg_emb, p=2, dim=-1)
            if neg_items.dim() == 3:
                neg_scores = torch.einsum("bsd,bsnd->bsn", hidden_states, neg_emb)
            else:  # 2D: [batch, num_neg, dim]
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

        # Negative-sampling BCE (word2vec / SASRec / loss-paper eq.):
        #   L = -log σ(pos) - (1/N) Σ_n log(1 - σ(neg_n))
        # The positive keeps FULL weight; negatives are AVERAGED. This must NOT
        # be a single mean over the combined [pos, neg×N] vector -- that would
        # dilute the positive term to weight 1/(1+N), so at large N the positive
        # signal vanishes and the model collapses (all-scores-down trivial
        # solution). Keeping the positive at weight 1 makes the loss balance
        # invariant to N, matching sampled-softmax's stability across N.
        bce = nn.functional.binary_cross_entropy_with_logits
        pos_loss = bce(
            pos_scores.squeeze(-1),
            torch.ones_like(pos_scores.squeeze(-1)),
            reduction="none",
        )  # [batch, seq_len]
        neg_loss = bce(
            neg_scores, torch.zeros_like(neg_scores), reduction="none"
        ).mean(dim=-1)  # [batch, seq_len]

        loss = (pos_loss + neg_loss) * mask
        return loss.sum() / mask.sum().clamp(min=1)
