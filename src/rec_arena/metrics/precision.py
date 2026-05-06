import torch
import numpy as np


def precision_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Universal Precision@K for all recommendation models."""
    if predictions.numel() == 0 or k == 0:
        return 0.0
    
    # Handle batch vs single prediction
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
    
    batch_size = predictions.size(0)
    precision_scores = []
    
    for i in range(batch_size):
        pred = predictions[i]
        
        # Convert targets to relevance scores
        if targets.dim() == 1:
            # Sequential case: single target item
            relevance = torch.zeros_like(pred)
            if targets[i] < len(relevance):
                relevance[targets[i]] = 1.0
        else:
            # Ranking case: relevance vector
            relevance = targets[i]
        
        _, top_k_indices = torch.topk(pred, min(k, len(pred)))
        relevant_retrieved = relevance[top_k_indices].sum().item()
        
        precision_scores.append(relevant_retrieved / min(k, len(pred)))
    
    return float(np.mean(precision_scores))