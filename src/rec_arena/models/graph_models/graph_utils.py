"""Utility functions for graph models."""
import torch
import pandas as pd
from typing import Dict, List


def create_edge_index(train_df: pd.DataFrame, num_users: int) -> torch.Tensor:
    """Create edge index from training data for graph models."""
    user_ids = train_df['user_id'].values
    item_ids = train_df['item_id'].values
    
    # Create bidirectional edges: user->item and item->user
    # Items are offset by num_users to create a unified node space
    user_to_item = torch.stack([
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids + num_users, dtype=torch.long)
    ])
    
    # Create reverse edges by flipping the rows
    item_to_user = torch.stack([
        user_to_item[1],  # item_ids + num_users (was second row, now first)
        user_to_item[0]   # user_ids (was first row, now second)
    ])
    
    # Concatenate both directions
    edge_index = torch.cat([user_to_item, item_to_user], dim=1)
    
    return edge_index


def add_graph_support_to_datamodule(datamodule_class):
    """Add graph support methods to RecDataModule without modifying the original class."""
    
    def _create_edge_index(self, train_df):
        """Create edge index from training data for graph models."""
        return create_edge_index(train_df, self.dataset.num_users)
    
    def get_edge_index(self):
        """Get edge index for graph models."""
        if not hasattr(self, 'edge_index') or self.edge_index is None:
            raise ValueError("Edge index not available. Make sure to call setup() first and use format='graph'")
        return self.edge_index
    
    def get_pyg_edge_index(self):
        """Get edge index in PyTorch Geometric format (user->item only, items offset by num_users)."""
        if not hasattr(self, 'edge_index') or self.edge_index is None:
            raise ValueError("Edge index not available. Make sure to call setup() first and use format='graph'")
        
        # Get only user->item edges (first half of the bidirectional edge_index)
        num_edges = self.edge_index.shape[1] // 2
        return self.edge_index[:, :num_edges]
    
    # Add methods to the class
    datamodule_class._create_edge_index = _create_edge_index
    datamodule_class.get_edge_index = get_edge_index
    datamodule_class.get_pyg_edge_index = get_pyg_edge_index
    
def create_pyg_edge_index(train_df: pd.DataFrame, num_users: int) -> torch.Tensor:
    """Create PyTorch Geometric compatible edge index (bidirectional, items offset)."""
    user_ids = train_df['user_id'].values
    item_ids = train_df['item_id'].values
    
    # Create bidirectional edges: user->item and item->user
    # Items are offset by num_users to create a unified node space
    user_to_item = torch.stack([
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids + num_users, dtype=torch.long)
    ])
    
    # Create reverse edges by flipping the rows
    item_to_user = torch.stack([
        user_to_item[1],  # item_ids + num_users (was second row, now first)
        user_to_item[0]   # user_ids (was first row, now second)
    ])
    
    # Concatenate both directions
    edge_index = torch.cat([user_to_item, item_to_user], dim=1)
    
    return edge_index


def add_pyg_support_to_datamodule(datamodule_class):
    """Add PyTorch Geometric support methods to RecDataModule."""
    
    def _create_pyg_edge_index(self, train_df):
        """Create PyG edge index from training data."""
        return create_pyg_edge_index(train_df, self.dataset.num_users)
    
    def get_pyg_edge_index(self):
        """Get PyG edge index."""
        if not hasattr(self, 'edge_index') or self.edge_index is None:
            raise ValueError("Edge index not available. Make sure to call setup() first and use format='pyg_graph'")
        return self.edge_index
    
    # Add methods to the class
    datamodule_class._create_pyg_edge_index = _create_pyg_edge_index
    datamodule_class.get_pyg_edge_index = get_pyg_edge_index
    
    return datamodule_class
