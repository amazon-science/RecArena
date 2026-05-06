"""LLaDA-style weighted cross-entropy loss."""

import torch
from torch import nn


class LLaDALoss(nn.Module):
    """Weighted cross-entropy loss for LLaDA-style diffusion models.
    
    Divides loss by masking ratio p_mask to create an upper bound on negative log-likelihood.
    """

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def __call__(self, logits=None, targets=None, mask=None, p_mask=None, 
                 hidden_states=None, item_embeddings=None, **kwargs):
        """Compute weighted cross-entropy loss.

        Args:
            logits: [batch, seq_len, vocab_size] (optional)
            targets: [batch, seq_len] (0-indexed)
            mask: [batch, seq_len] - which positions were masked
            p_mask: [batch, seq_len] - masking ratio per position (optional, defaults to 1.0)
            hidden_states: [batch, seq_len, dim] (optional)
            item_embeddings: [vocab_size, dim] (optional)

        Returns:
            Scalar loss value
        """
        if logits is None:
            if hidden_states is not None and item_embeddings is not None:
                logits = torch.matmul(hidden_states, item_embeddings.transpose(0, 1))
            else:
                raise ValueError("Must provide either logits or (hidden_states + item_embeddings)")
        
        # Default p_mask to 1.0 if not provided (standard CE)
        if p_mask is None:
            p_mask = torch.ones_like(targets, dtype=logits.dtype)
        
        # Flatten
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)
        p_mask_flat = p_mask.reshape(-1)
        
        # Compute weighted loss: CE / p_mask
        loss = self.cross_entropy(logits_flat, targets_flat) / p_mask_flat.clamp(min=1e-8)
        loss = loss * mask_flat.float()
        
        # Average over sequence length (not batch size)
        return loss.sum() / (logits.size(0) * logits.size(1))
