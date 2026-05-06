import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization."""
    
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] or [batch_size, seq_len, num_classes]
            targets: [batch_size] or [batch_size, seq_len]
        """
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
        
        num_classes = logits.size(-1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        
        # Set true class probability
        mask = targets != self.ignore_index
        smooth_targets[mask, targets[mask]] = 1.0 - self.smoothing
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        return loss[mask].mean() if mask.any() else loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for imbalanced data."""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] or [batch_size, seq_len, num_classes]
            targets: [batch_size] or [batch_size, seq_len]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive learning loss for sequential recommendation."""
    
    def __init__(self, temperature=0.1, negative_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.negative_weight = negative_weight
    
    def forward(self, anchor_embs, positive_embs, negative_embs=None):
        """
        Args:
            anchor_embs: [batch_size, emb_dim] - sequence representations
            positive_embs: [batch_size, emb_dim] - positive item embeddings
            negative_embs: [batch_size, num_negatives, emb_dim] - negative item embeddings
        """
        # Normalize embeddings
        anchor_embs = F.normalize(anchor_embs, dim=-1)
        positive_embs = F.normalize(positive_embs, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(anchor_embs * positive_embs, dim=-1) / self.temperature
        
        if negative_embs is not None:
            # Negative similarities
            negative_embs = F.normalize(negative_embs, dim=-1)
            neg_sim = torch.bmm(
                anchor_embs.unsqueeze(1), 
                negative_embs.transpose(1, 2)
            ).squeeze(1) / self.temperature
            
            # Contrastive loss with negatives
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            
            return F.cross_entropy(logits, targets)
        else:
            # Simple contrastive loss (maximize positive similarity)
            return -pos_sim.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining next-item prediction and rating prediction."""
    
    def __init__(self, next_item_weight=1.0, rating_weight=0.5, rating_loss='mse'):
        super().__init__()
        self.next_item_weight = next_item_weight
        self.rating_weight = rating_weight
        self.rating_loss = rating_loss
        
        if rating_loss == 'mse':
            self.rating_criterion = nn.MSELoss()
        elif rating_loss == 'mae':
            self.rating_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported rating loss: {rating_loss}")
    
    def forward(self, next_item_logits, next_item_targets, rating_preds, rating_targets, mask=None):
        """
        Args:
            next_item_logits: [batch_size, seq_len, num_items] - next item predictions
            next_item_targets: [batch_size, seq_len] - next item targets
            rating_preds: [batch_size, seq_len] - rating predictions
            rating_targets: [batch_size, seq_len] - rating targets
            mask: [batch_size, seq_len] - padding mask
        """
        # Next item prediction loss
        next_item_loss = F.cross_entropy(
            next_item_logits.view(-1, next_item_logits.size(-1)),
            next_item_targets.view(-1),
            reduction='none'
        ).view(next_item_targets.shape)
        
        # Rating prediction loss
        rating_loss = self.rating_criterion(rating_preds, rating_targets)
        if rating_loss.dim() == 0:  # If scalar, expand to match shape
            rating_loss = rating_loss.expand_as(rating_targets)
        else:
            rating_loss = (rating_preds - rating_targets) ** 2  # MSE per element
        
        # Apply mask if provided
        if mask is not None:
            next_item_loss = next_item_loss * mask
            rating_loss = rating_loss * mask
            
            next_item_loss = next_item_loss.sum() / mask.sum()
            rating_loss = rating_loss.sum() / mask.sum()
        else:
            next_item_loss = next_item_loss.mean()
            rating_loss = rating_loss.mean()
        
        # Combine losses
        total_loss = (
            self.next_item_weight * next_item_loss + 
            self.rating_weight * rating_loss
        )
        
        return {
            'total_loss': total_loss,
            'next_item_loss': next_item_loss,
            'rating_loss': rating_loss
        }