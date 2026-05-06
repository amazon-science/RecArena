import torch
from typing import Dict, List, Callable
from .ndcg import ndcg_at_k
from .recall import recall_at_k
from .precision import precision_at_k
from .hit_rate import hit_rate_at_k
from .mrr import mrr_at_k


class MetricCalculator:
    """Universal metric calculator for all recommendation models."""
    
    def __init__(self, k_values: List[int] = None):
        if k_values is None:
            k_values = [5, 10, 20]
        self.k_values = k_values
        
        # Universal metrics that work for all models
        self.metrics = {
            'ndcg': ndcg_at_k,
            'recall': recall_at_k,
            'precision': precision_at_k,
            'hit_rate': hit_rate_at_k,
            'mrr': mrr_at_k,
        }
    
    def calculate_all(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate all metrics for given predictions and targets.
        
        Args:
            predictions: [batch_size, num_items] or [num_items] prediction scores
            targets: [batch_size] target IDs (sequential) OR [batch_size, num_items] relevance (ranking)
        """
        results = {}
        
        for metric_name, metric_fn in self.metrics.items():
            for k in self.k_values:
                key = f"{metric_name}@{k}"
                results[key] = metric_fn(predictions, targets, k)
        
        return results
    
    def add_metric(self, name: str, metric_fn: Callable) -> None:
        """Add a custom metric function."""
        if not callable(metric_fn):
            raise ValueError("metric_fn must be callable")
        self.metrics[name] = metric_fn
    
    def set_k_values(self, k_values: List[int]) -> None:
        """Update k values for evaluation."""
        if not all(isinstance(k, int) and k > 0 for k in k_values):
            raise ValueError("All k_values must be positive integers")
        self.k_values = k_values