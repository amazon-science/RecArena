"""Reproducibility utilities for RecArena."""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get appropriate device for computation.
    
    Args:
        device: Specific device to use ('cpu', 'cuda', 'mps')
        
    Returns:
        PyTorch device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class ReproducibilityContext:
    """Context manager for reproducible operations."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.original_state = None
    
    def __enter__(self):
        # Save current state
        self.original_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            self.original_state['torch_cuda'] = torch.cuda.get_rng_state()
        
        # Set seed
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        random.setstate(self.original_state['python'])
        np.random.set_state(self.original_state['numpy'])
        torch.set_rng_state(self.original_state['torch'])
        if 'torch_cuda' in self.original_state:
            torch.cuda.set_rng_state(self.original_state['torch_cuda'])