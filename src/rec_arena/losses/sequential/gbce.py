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

    def __init__(
        self,
        alpha: float = None,
        t: float = 0.5,
        eps: float = 1e-10,
        num_negatives: int = None,
        vocab_size: int = None,
    ) -> None:
        """
        Args:
            alpha: float in (0, 1] representing the negative sampling rate. If
                explicitly provided it is honored as-is (backward compatible). If
                left as None, alpha is auto-calibrated from num_negatives /
                (vocab_size - 1) when both are given, else defaults to 0.5.
            t: float in [0, 1] temperature parameter
            eps: float >=0 used to define bounds for numerical stability
            num_negatives: number of negative samples per positive. Used together
                with vocab_size to auto-calibrate alpha.
            vocab_size: catalog / vocabulary size N. Used together with
                num_negatives to auto-calibrate alpha.

        Alpha calibration (gSASRec paper, https://arxiv.org/pdf/2308.07192):
        alpha must equal the negative sampling rate ~= num_negatives / (N - 1),
        NOT a static 0.5. With e.g. 128 negatives over a 10k+ catalog the true
        alpha ~= 0.01, so a hard-coded 0.5 makes the beta calibration term (and
        the p^{-beta} logit transform in forward()) materially wrong. We only
        auto-calibrate when the caller did not pass an explicit alpha.
        """
        super().__init__()
        # Only auto-calibrate when alpha was not explicitly supplied.
        if alpha is None:
            if num_negatives is not None and vocab_size is not None and vocab_size > 1:
                alpha = num_negatives / (vocab_size - 1)
                # Clamp to a safe open interval so beta stays finite and the
                # p^{-beta} path in forward() cannot blow up / divide by zero.
                alpha = max(1e-6, min(alpha, 1.0))
            else:
                # Fall back to the historical default so nothing crashes.
                alpha = 0.5
        assert 0 <= t <= 1
        assert 0 < alpha <= 1
        assert eps > 0
        self.alpha = alpha
        self.t = t
        self.eps = eps
        # beta = alpha*(t*(1 - 1/alpha) + 1/alpha). For small alpha (~0.01) beta
        # stays in (0, 1], so p^{-beta} is a mild power and the existing clamps
        # in forward() keep it numerically safe (float64 + finfo.max ceiling).
        self.beta = alpha * (t * (1 - 1.0 / alpha) + 1 / alpha)

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
        """Compute GBCE loss with negative sampling.

        Args:
            logits: [batch, seq_len, vocab_size] (optional, for backward compatibility)
            targets: [batch, seq_len] (0-indexed)
            mask: [batch, seq_len]
            neg_items: [batch, seq_len, num_neg] or [batch, num_neg] (required)
            hidden_states: [batch, seq_len, dim] (optional, for fast path)
            item_embeddings: [vocab_size, dim] (optional, for fast path)
            embedding_lookup: optional ids->emb callable for sparse-safe lookup.

        Returns:
            Scalar loss value
        """
        if neg_items is None:
            raise ValueError(
                "GBCE loss requires negative samples in neg_items parameter"
            )

        batch_size, seq_len = (
            targets.shape if targets.dim() == 2 else (targets.shape[0], 1)
        )

        # Fast path: compute sampled logits directly from hidden states
        if hidden_states is not None and item_embeddings is not None:
            lookup = (
                embedding_lookup
                if embedding_lookup is not None
                else (lambda ids: item_embeddings[ids])
            )
            # Gather embeddings for positive and negative items
            pos_emb = lookup(targets)  # [batch, seq_len, dim]

            # Compute positive scores
            pos_scores = (hidden_states * pos_emb).sum(
                dim=-1, keepdim=True
            )  # [batch, seq_len, 1]

            # Handle negative items (2D or 3D)
            neg_emb = lookup(neg_items)
            if neg_items.dim() == 3:
                # Per-position negatives: [batch, seq_len, num_neg, dim]
                neg_scores = torch.einsum(
                    "bsd,bsnd->bsn", hidden_states, neg_emb
                )  # [batch, seq_len, num_neg]
            elif neg_items.dim() == 2:
                # Global negatives: [batch, num_neg, dim]
                neg_scores = torch.einsum(
                    "bsd,bnd->bsn", hidden_states, neg_emb
                )  # [batch, seq_len, num_neg]
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

        # Negative-sampling BCE structure (see BCENegativeSamplingLoss): the
        # positive term keeps FULL weight, negatives are AVERAGED. A single
        # mean over the combined [pos, neg×N] vector dilutes the positive to
        # weight 1/(1+N), so at large N gBCE collapses to the trivial
        # all-scores-down solution (empirically dead by N>=512). The gBCE
        # calibration lives in the transformed positive logit above; the loss
        # weighting must still keep the positive at weight 1.
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        pos_loss = bce(
            positive_logits_transformed.squeeze(-1),
            torch.ones_like(positive_logits_transformed.squeeze(-1)),
            reduction="none",
        )  # [batch, seq_len]
        neg_loss = bce(
            negative_logits,
            torch.zeros_like(negative_logits),
            reduction="none",
        ).mean(-1)  # [batch, seq_len]

        loss_per_element = (pos_loss + neg_loss) * mask
        # clamp(min=1) guards against an all-zero mask (empty batch) -> div by 0.
        return loss_per_element.sum() / mask.sum().clamp(min=1)
