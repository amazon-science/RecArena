import torch
import numpy as np


def recall_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Universal Recall@K for all recommendation models."""
    if predictions.numel() == 0:
        return 0.0
    
    # Handle batch vs single prediction
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
    
    batch_size = predictions.size(0)
    recall_scores = []
    
    num_items = predictions.size(1)
    for i in range(batch_size):
        pred = predictions[i]
        
        # Convert targets to relevance scores
        if targets.dim() == 1:
            # Sequential case: single target item (starts at 3)
            relevance = torch.zeros_like(pred)
            target_idx = targets[i].item()
            if 3 <= target_idx < num_items + 3:
                relevance[target_idx] = 1.0
        else:
            # Ranking case: relevance vector
            relevance = targets[i]
        
        if relevance.sum() == 0:
            recall_scores.append(0.0)
            continue
        
        _, top_k_indices = torch.topk(pred, min(k, len(pred)))
        relevant_retrieved = relevance[top_k_indices].sum().item()
        total_relevant = relevance.sum().item()
        
        recall_scores.append(relevant_retrieved / total_relevant)
    
    return float(np.mean(recall_scores))