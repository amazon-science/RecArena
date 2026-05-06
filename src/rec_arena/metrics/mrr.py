import torch
import numpy as np


def mrr_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Universal Mean Reciprocal Rank@K."""
    if predictions.numel() == 0:
        return 0.0
    
    # Handle batch vs single prediction
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
    
    batch_size = predictions.size(0)
    reciprocal_ranks = []
    
    for i in range(batch_size):
        pred = predictions[i]
        _, top_k_indices = torch.topk(pred, min(k, len(pred)))
        
        if targets.dim() == 1:
            # Sequential case: find rank of target item
            positions = (top_k_indices == targets[i]).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                rank = positions[0].item() + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            # Ranking case: find rank of first relevant item
            relevance = targets[i]
            for j, idx in enumerate(top_k_indices):
                if relevance[idx] > 0:
                    reciprocal_ranks.append(1.0 / (j + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks))