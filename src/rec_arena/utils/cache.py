"""Caching utilities for RecArena."""

import torch
from functools import lru_cache
from typing import Dict, Any, Optional


class TensorCache:
    """Simple tensor cache for frequently used computations."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Put tensor in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = tensor.detach().clone()
        self.access_count[key] = 1
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()


# Global cache instance
_tensor_cache = TensorCache()


def cached_computation(key: str):
    """Decorator for caching tensor computations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cached_result = _tensor_cache.get(key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                _tensor_cache.put(key, result)
            return result
        return wrapper
    return decorator