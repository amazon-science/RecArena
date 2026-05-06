"""Data augmentation for sequential recommendation."""
import random
from typing import List


class SequenceAugmenter:
    """Augment sequences for better generalization."""
    
    def __init__(self, crop_prob: float = 0.3, mask_prob: float = 0.2, reorder_prob: float = 0.1):
        self.crop_prob = crop_prob
        self.mask_prob = mask_prob
        self.reorder_prob = reorder_prob
    
    def crop(self, sequence: List[int], min_len: int = 2) -> List[int]:
        """Randomly crop sequence."""
        if len(sequence) <= min_len or random.random() > self.crop_prob:
            return sequence
        crop_len = random.randint(min_len, len(sequence))
        start = random.randint(0, len(sequence) - crop_len)
        return sequence[start:start + crop_len]
    
    def mask(self, sequence: List[int], mask_token: int = 2) -> List[int]:
        """Randomly mask items."""
        if random.random() > self.mask_prob:
            return sequence
        masked = sequence.copy()
        num_mask = max(1, int(len(sequence) * 0.15))
        mask_indices = random.sample(range(len(sequence)), num_mask)
        for idx in mask_indices:
            masked[idx] = mask_token
        return masked
    
    def reorder(self, sequence: List[int], window: int = 3) -> List[int]:
        """Randomly reorder within small windows."""
        if len(sequence) <= window or random.random() > self.reorder_prob:
            return sequence
        reordered = sequence.copy()
        for i in range(0, len(sequence) - window + 1, window):
            chunk = reordered[i:i + window]
            random.shuffle(chunk)
            reordered[i:i + window] = chunk
        return reordered
    
    def augment(self, sequence: List[int]) -> List[int]:
        """Apply random augmentation."""
        aug_type = random.choice(['crop', 'mask', 'reorder', 'none'])
        if aug_type == 'crop':
            return self.crop(sequence)
        elif aug_type == 'mask':
            return self.mask(sequence)
        elif aug_type == 'reorder':
            return self.reorder(sequence)
        return sequence
