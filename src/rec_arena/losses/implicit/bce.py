import torch
from torch import nn


class BCELoss(nn.Module):
    """BCE loss for implicit feedback models with negative sampling.
    
    Unified interface requirements:
    - model.get_hidden_states(user_ids, item_ids) -> hidden representation
    - model.prediction(hidden_states) -> score tensor
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def __call__(self, model, batch, hidden_states):
        """BCE loss with negative sampling: positive item gets label=1, negatives get label=0."""
        user_ids = batch["user_id"]
        neg_items = batch.get("neg_items")
        
        if neg_items is None:
            raise ValueError("BCE loss requires negative samples in batch['neg_items']")
        
        batch_size, num_neg = neg_items.size()
        
        # Positive predictions (label=1)
        pos_predictions = model.prediction(hidden_states).squeeze(-1)
        pos_labels = torch.ones_like(pos_predictions)
        
        # Negative predictions (label=0)
        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_neg)
        neg_items_clamped = torch.clamp(neg_items, 0, model.config.num_items - 1)
        neg_hidden_states = model.get_hidden_states(user_ids_expanded.reshape(-1), neg_items_clamped.reshape(-1))
        neg_predictions = model.prediction(neg_hidden_states).squeeze(-1).view(batch_size, num_neg)
        neg_labels = torch.zeros_like(neg_predictions)
        
        # Combine positive and negative
        all_predictions = torch.cat([pos_predictions.unsqueeze(1), neg_predictions], dim=1).flatten()
        all_labels = torch.cat([pos_labels.unsqueeze(1), neg_labels], dim=1).flatten()
        
        return self.bce_loss(all_predictions, all_labels)