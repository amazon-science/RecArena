"""Memory optimization utilities."""

import torch
import gc
from contextlib import contextmanager


@contextmanager
def memory_efficient_mode():
    """Context manager for memory efficient operations."""
    # Save current settings
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic
    
    try:
        # Enable memory efficient settings
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield
        
    finally:
        # Restore settings
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def optimize_model_memory(model):
    """Apply memory optimizations to model."""
    # Convert to half precision if possible
    if torch.cuda.is_available():
        model = model.half()
    
    # Enable gradient checkpointing for large models
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()