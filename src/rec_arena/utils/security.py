"""Security utilities for RecArena."""

import os
from pathlib import Path
from typing import Union


def validate_path(path: Union[str, Path], allowed_extensions: list = None) -> Path:
    """Validate file path for security.
    
    Args:
        path: File path to validate
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or unsafe
    """
    if not path:
        raise ValueError("Path cannot be empty")
    
    path_obj = Path(path).resolve()
    
    # Check for path traversal
    if not str(path_obj).startswith(os.getcwd()):
        raise ValueError("Path must be within current directory")
    
    # Check for suspicious patterns
    if ".." in str(path_obj):
        raise ValueError("Path traversal detected")
    
    # Validate file extension
    if allowed_extensions and not any(str(path_obj).endswith(ext) for ext in allowed_extensions):
        raise ValueError(f"File extension must be one of: {allowed_extensions}")
    
    return path_obj


def safe_torch_load(path: Union[str, Path], **kwargs):
    """Safely load PyTorch model with security checks.
    
    Args:
        path: Path to model file
        **kwargs: Additional arguments for torch.load
        
    Returns:
        Loaded model state dict
    """
    import torch
    
    validated_path = validate_path(path, ['.pt', '.pth'])
    
    if not validated_path.exists():
        raise FileNotFoundError(f"Model file not found: {validated_path}")
    
    # Use weights_only=True for security
    kwargs.setdefault('weights_only', True)
    kwargs.setdefault('map_location', 'cpu')
    
    return torch.load(validated_path, **kwargs)