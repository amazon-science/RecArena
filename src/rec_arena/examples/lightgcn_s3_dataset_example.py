"""LightGCN example using RecMDataset to load pre-split data from S3."""
import os
import torch
from lightning import Trainer
from rec_arena.datasets import RecMDataset, RecDataModule
from rec_arena.models import PyGLightGCN
from rec_arena.configs.defaults.lightgcn import LightGCNConfig
from rec_arena.samplers import GraphRandomSampler
from rec_arena.metrics import MetricCalculator

# Set AWS profile
os.environ['AWS_PROFILE'] = 'example-account'

def main():
    # Load pre-split dataset from S3
    dataset = RecMDataset(
        dataset_name="ml_100k",
        split_type="leave_one_out",
        s3_bucket="example-bucket",
        s3_prefix="recarena"
    )
    dataset.load_data()
    
    # Convert explicit ratings to implicit feedback (rating >= 4 becomes positive)
    dataset.train_df['implicit'] = (dataset.train_df['rating'] >= 4).astype(int)
    dataset.test_df['implicit'] = (dataset.test_df['rating'] >= 4).astype(int)
    
    # Remap user and item IDs to be contiguous (0-based)
    all_users = sorted(set(dataset.train_df['user_id'].unique()) | set(dataset.test_df['user_id'].unique()))
    all_items = sorted(set(dataset.train_df['item_id'].unique()) | set(dataset.test_df['item_id'].unique()))
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(all_users)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(all_items)}
    
    dataset.train_df['user_id'] = dataset.train_df['user_id'].map(user_map)
    dataset.train_df['item_id'] = dataset.train_df['item_id'].map(item_map)
    dataset.test_df['user_id'] = dataset.test_df['user_id'].map(user_map)
    dataset.test_df['item_id'] = dataset.test_df['item_id'].map(item_map)
    
    # Update dataset counts
    dataset.num_users = len(all_users)
    dataset.num_items = len(all_items)
    
    print(f"Dataset loaded: {dataset.num_users} users, {dataset.num_items} items")
    
    # Create negative sampler for BPR loss
    negative_sampler = GraphRandomSampler(num_items=dataset.num_items, num_negatives=1, seed=42)
    
    # Create datamodule for graph format
    datamodule = RecDataModule(
        dataset,
        format="graph",
        batch_size=512,
        num_workers=0,
        negative_sampler=negative_sampler
    )
    datamodule.setup()
    
    # Configure model
    config = LightGCNConfig(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=64,
        num_layers=3,
        lr=1e-3,
        weight_decay=1e-4,
        compute_val_metrics=True,
        val_k_values=[10]
    )
    
    # Create model and set graph data
    model = PyGLightGCN(config)
    edge_index = datamodule.get_edge_index()
    model.set_graph_data(edge_index)
    
    print("\n=== Training LightGCN ===")
    
    # Train
    trainer = Trainer(
        max_epochs=10,
        accelerator="cpu",
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(model, datamodule)
    
    # Evaluate on test set
    print("\n=== Evaluating LightGCN ===")
    
    model.eval()
    test_loader = datamodule.test_dataloader()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            result = model.test_step(batch, 0)
            all_predictions.append(result["predictions"])
            all_targets.append(result["targets"])
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    calculator = MetricCalculator(k_values=[10])
    results = calculator.calculate_all(all_predictions, all_targets)
    
    print(f"NDCG@10: {results['ndcg@10']:.4f}")
    print(f"Hit Rate@10: {results['hit_rate@10']:.4f}")
    print(f"Precision@10: {results['precision@10']:.4f}")
    print(f"Recall@10: {results['recall@10']:.4f}")

if __name__ == "__main__":
    main()
