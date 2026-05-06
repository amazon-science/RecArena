"""Performance optimization utilities."""

import torch
import time
from functools import wraps
from typing import Callable, Any
from contextlib import contextmanager


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@contextmanager
def torch_profile(enabled: bool = True):
    """Context manager for PyTorch profiling."""
    if not enabled:
        yield None
        return
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        yield prof


def optimize_tensor_ops():
    """Apply tensor operation optimizations."""
    # Enable TensorFloat-32 for faster training on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        pass


class MemoryTracker:
    """Track GPU memory usage."""
    
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
    
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
    
    def stop(self):
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()
            return {
                'peak_memory_mb': self.peak_memory / 1024**2,
                'current_memory_mb': current_memory / 1024**2,
                'memory_increase_mb': (current_memory - self.start_memory) / 1024**2
            }
        return {'peak_memory_mb': 0, 'current_memory_mb': 0, 'memory_increase_mb': 0}