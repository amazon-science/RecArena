"""Dataset constants and special tokens.

RecArena uses 0-based item indexing throughout:
- Item IDs: [0, 1, 2, ..., num_items-1]
- Special tokens are reserved at the beginning of the vocabulary
"""

# Special tokens (0-based indexing)
PAD_TOKEN = 0  # Padding token for sequences
MASK_TOKEN = 1  # Mask token for BERT4Rec
UNK_TOKEN = 2  # Unknown token (reserved for future use)

# First valid item ID (after special tokens)
FIRST_ITEM_ID = 3
