import torch
import numpy as np


def hit_rate_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Universal Hit Rate@K - whether target is in top-k."""
    if predictions.numel() == 0:
        return 0.0
    
    # Handle batch vs single prediction
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
    
    batch_size = predictions.size(0)
    hits = []
    
    for i in range(batch_size):
        pred = predictions[i]
        
        _, top_k_indices = torch.topk(pred, min(k, len(pred)))
        
        if targets.dim() == 1:
            # Sequential case: check if target item is in top-k
            hit = (top_k_indices == targets[i]).any().float().item()
        else:
            # Ranking case: check if any relevant item is in top-k
            relevance = targets[i]
            hit = (relevance[top_k_indices] > 0).any().float().item()
        
        hits.append(hit)
    
    return float(np.mean(hits))