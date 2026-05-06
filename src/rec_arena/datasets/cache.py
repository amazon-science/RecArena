"""Dataset caching utilities."""
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional


class DatasetCache:
    """Cache processed datasets to disk."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, **kwargs) -> Optional[Any]:
        """Retrieve cached data."""
        cache_key = self._get_cache_key(**kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, data: Any, **kwargs) -> None:
        """Cache data to disk."""
        cache_key = self._get_cache_key(**kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
