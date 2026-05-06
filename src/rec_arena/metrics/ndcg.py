import torch


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Universal NDCG@K for all recommendation models (vectorized).
    
    Args:
        predictions: [batch_size, num_items] or [num_items] prediction scores
        targets: [batch_size] target item IDs (1-indexed) OR [batch_size, num_items] relevance scores
        k: cutoff value
        
    Returns:
        NDCG@K score
    """
    if predictions.numel() == 0:
        return 0.0
    
    # Handle batch vs single prediction
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
    
    batch_size, num_items = predictions.size()
    k = min(k, num_items)
    
    # Convert targets to relevance scores
    if targets.dim() == 1:
        # Sequential case: targets[i] is single item ID (starts at 3 for GPT-style)
        relevance = torch.zeros_like(predictions)
        valid_mask = (targets >= 0) & (targets < predictions.size(1))
        relevance[torch.arange(batch_size)[valid_mask], targets[valid_mask]] = 1.0
    else:
        # Ranking case: targets[i] is relevance vector
        relevance = targets
    
    # Get top-k predictions
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    
    # Gather relevance scores for top-k items
    top_k_relevance = torch.gather(relevance, 1, top_k_indices)
    
    # Calculate DCG (vectorized)
    positions = torch.arange(2, k + 2, dtype=torch.float32, device=predictions.device)
    dcg = (top_k_relevance / torch.log2(positions)).sum(dim=-1)
    
    # Calculate IDCG (vectorized)
    sorted_relevance, _ = torch.sort(relevance, descending=True, dim=-1)
    idcg = (sorted_relevance[:, :k] / torch.log2(positions)).sum(dim=-1)
    
    # Compute NDCG
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    
    return ndcg.mean().item()