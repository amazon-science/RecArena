"""Unified tokenizer for recommendation data."""

from typing import List, Dict, Optional
import numpy as np


class RecTokenizer:
    """HuggingFace-style tokenizer for recommendation data."""
    
    def __init__(self):
        self.vocab_size = None
        self.item_to_token = {}
        self.token_to_item = {}
        
        # Special tokens
        self.pad_token = 0
        self.unk_token = 1
        self.mask_token = 2
        self.special_tokens = {'[PAD]': 0, '[UNK]': 1, '[MASK]': 2}
    
    def fit(self, sequences: List[List[int]]):
        """Fit tokenizer on sequences (HuggingFace style)."""
        # Get all unique items
        all_items = set()
        for seq in sequences:
            all_items.update(seq)
        
        # Create vocabulary (starting from 3 to reserve special tokens)
        sorted_items = sorted(all_items)
        self.item_to_token = {item: idx + 3 for idx, item in enumerate(sorted_items)}
        self.token_to_item = {idx + 3: item for idx, item in enumerate(sorted_items)}
        
        # Add special tokens to mappings
        for token, idx in self.special_tokens.items():
            self.token_to_item[idx] = token
        
        self.vocab_size = len(sorted_items) + 3  # +3 for special tokens
    
    def encode(self, sequence: List[int]) -> List[int]:
        """Encode sequence to tokens."""
        return [self.item_to_token.get(item, self.unk_token) for item in sequence]
    
    def decode(self, tokens: List[int]) -> List[int]:
        """Decode tokens back to items."""
        decoded = []
        for token in tokens:
            if token in self.token_to_item and token >= 3:  # Skip special tokens
                decoded.append(self.token_to_item[token])
        return decoded
    
    def encode_batch(self, sequences: List[List[int]]) -> List[List[int]]:
        """Encode batch of sequences."""
        return [self.encode(seq) for seq in sequences]
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int) -> List[List[int]]:
        """Pad sequences to max_length."""
        padded = []
        for seq in sequences:
            if len(seq) >= max_length:
                padded.append(seq[:max_length])
            else:
                padded.append(seq + [self.pad_token] * (max_length - len(seq)))
        return padded
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size