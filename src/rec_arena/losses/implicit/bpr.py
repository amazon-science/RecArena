import torch
import torch.nn.functional as F
from torch import nn


class BPRLoss(nn.Module):
    """BPR loss for implicit feedback models with negative sampling.

    Unified interface requirements:
    - model.get_hidden_states(user_ids, item_ids) -> hidden representation
    - model.prediction(hidden_states) -> score tensor
    """

    def __init__(self):
        super().__init__()

    def __call__(self, model, batch, hidden_states):
        """BPR loss for user-item ranking with L2 regularization."""
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]  # 0-indexed
        neg_items = batch.get("neg_items")  # 0-indexed

        if neg_items is None:
            raise ValueError("BPR loss requires negative samples in batch['neg_items']")

        # Get positive scores
        pos_predictions = model.prediction(hidden_states).squeeze(-1)  # [batch]

        # OPTIMIZED: Get negative scores efficiently for matrix factorization models
        batch_size, num_neg = neg_items.size()

        # Clamp to valid range as safety check [0, num_items-1]
        neg_items_clamped = torch.clamp(neg_items, 0, model.config.num_items - 1)

        # Check if model has direct embedding access (much more efficient!)
        if hasattr(model, "get_user_embedding") and hasattr(
            model, "get_item_embedding"
        ):
            # Efficient path: get embeddings directly and compute scores
            user_embs = model.get_user_embedding(user_ids)  # [batch, dim]
            neg_item_embs = model.get_item_embedding(
                neg_items_clamped
            )  # [batch, num_neg, dim]

            # Compute scores: broadcast user_embs over num_neg dimension
            neg_predictions = (user_embs.unsqueeze(1) * neg_item_embs).sum(
                dim=-1
            )  # [batch, num_neg]
        else:
            # Fallback: use get_hidden_states (slower, for complex models)
            user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_neg)
            neg_hidden_states = model.get_hidden_states(
                user_ids_expanded.reshape(-1), neg_items_clamped.reshape(-1)
            )
            neg_predictions = model.prediction(neg_hidden_states).squeeze(-1)
            neg_predictions = neg_predictions.view(batch_size, num_neg)

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        # Use F.logsigmoid for numerical stability instead of log(sigmoid())
        pos_scores = pos_predictions.unsqueeze(1)  # [batch, 1]
        diff = pos_scores - neg_predictions  # [batch, num_neg]
        bpr_loss = -F.logsigmoid(diff).mean()

        # Add L2 regularization on embeddings (critical for BPR-MF convergence!)
        # OPTIMIZED: Use squared norms instead of computing full norms
        reg_loss = 0.0
        if hasattr(model.config, "reg_weight") and model.config.reg_weight > 0:
            # Get the embeddings that were used in this batch
            user_embs = model.get_user_embedding(user_ids)
            pos_item_embs = model.get_item_embedding(item_ids)
            
            # L2 regularization: mean of squared norms
            # Only regularize user and positive item (standard BPR approach)
            # Regularizing all negatives is too expensive and not standard
            reg_loss = model.config.reg_weight * (
                (user_embs ** 2).sum() + (pos_item_embs ** 2).sum()
            ) / batch_size

        return bpr_loss + reg_loss
