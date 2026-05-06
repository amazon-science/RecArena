import torch
from torch import nn


class GBCE(nn.Module):
    """Pure Generalized Binary Cross-Entropy loss function.

    A stateless computational function from the gSASRec paper:
    https://arxiv.org/pdf/2308.07192 (eq. 8, 27)

    Expects pre-shifted and pre-processed inputs:
    - logits: [batch, seq_len, vocab_size] (pre-shifted, pre-processed) OR
    - hidden_states + item_embeddings (for fast sampled logits)
    - targets: [batch, seq_len] (pre-shifted, 0-indexed)
    - mask: [batch, seq_len] (pre-shifted)
    - neg_items: [batch, seq_len, num_neg] or [batch, num_neg] (required)
    """

    def __init__(self, alpha: float = 0.5, t: float = 0.5, eps: float = 1e-10) -> None:
        """
        Args:
            alpha: float in (0, 1] representing the negative sampling rate
            t: float in [0, 1] temperature parameter
            eps: float >=0 used to define bounds for numerical stability
        """
        super().__init__()
        assert 0 <= t <= 1
        assert 0 < alpha <= 1
        assert eps > 0
        self.alpha = alpha
        self.t = t
        self.eps = eps
        self.beta = alpha * (t * (1 - 1.0 / alpha) + 1 / alpha)

    def __call__(self, logits=None, targets=None, mask=None, neg_items=None,
                 hidden_states=None, item_embeddings=None):
        """Compute GBCE loss with negative sampling.

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
                "GBCE loss requires negative samples in neg_items parameter"
            )

        batch_size, seq_len = targets.shape if targets.dim() == 2 else (targets.shape[0], 1)

        # Fast path: compute sampled logits directly from hidden states
        if hidden_states is not None and item_embeddings is not None:
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
            
            # Combine: [batch, seq_len, 1 + num_neg]
            combined_logits = torch.cat([pos_scores, neg_scores], dim=-1)
        
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
            combined_logits = torch.cat([pos_scores, neg_scores], dim=-1)
        else:
            raise ValueError(
                "Must provide either (hidden_states + item_embeddings) or logits"
            )

        # Apply GBCE transformation
        return self.forward(combined_logits, mask)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Core GBCE computation.

        Args:
            logits: [batch, seq_len, num_neg + 1] with positive at [:, :, 0]
            mask: [batch, seq_len]

        Returns:
            Scalar loss value
        """
        positive_logits = logits[:, :, 0:1].to(torch.float64)
        negative_logits = logits[:, :, 1:].to(torch.float64)

        # Create labels: 1 for positive, 0 for negatives
        pos_neg_label = torch.zeros_like(logits, dtype=torch.float64)
        pos_neg_label[:, :, 0] = 1.0

        # Apply GBCE transformation to positive logits
        positive_probs = torch.clamp(
            torch.sigmoid(positive_logits), self.eps, 1 - self.eps
        )
        positive_probs_adjusted = torch.clamp(
            positive_probs.pow(-self.beta),
            1 + self.eps,
            torch.finfo(torch.float64).max,
        )

        to_log = torch.clamp(
            torch.div(1.0, (positive_probs_adjusted - 1)),
            self.eps,
            torch.finfo(torch.float64).max,
        )

        positive_logits_transformed = to_log.log()
        logits_transformed = torch.cat(
            [positive_logits_transformed, negative_logits], -1
        )

        # Compute BCE loss
        loss_per_element = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=logits_transformed, target=pos_neg_label, reduction="none"
            ).mean(-1)
            * mask
        )

        return loss_per_element.sum() / mask.sum()
