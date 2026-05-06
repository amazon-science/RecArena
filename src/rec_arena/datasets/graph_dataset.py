import torch
import pandas as pd
from typing import Dict, List, Set
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    """Dataset for graph neural network models."""
    
    def __init__(self, interactions: List[Dict], edge_index: torch.Tensor):
        self.interactions = interactions
        self.edge_index = edge_index
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        result = {
            'user_id': torch.tensor(interaction['user_id'], dtype=torch.long),
            'item_id': torch.tensor(interaction['item_id'], dtype=torch.long),
            'label': torch.tensor(interaction['label'], dtype=torch.float),
            'edge_index': self.edge_index
        }
        
        return result
    
def to_graph(df: pd.DataFrame) -> List[Dict]:
    """Convert to graph format for graph neural networks."""
    interactions = []
    
    # Positive interactions
    positive_df = df[df['implicit'] == 1]
    for _, row in positive_df.iterrows():
        interaction = {
            'user_id': int(row['user_id']),
            'item_id': int(row['item_id']),
            'label': 1
        }        
        interactions.append(interaction)
    
    return interactions