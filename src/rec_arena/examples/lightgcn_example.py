"""PyG LightGCN example with RecArena framework."""

import torch
import os
import urllib.request
import zipfile
import ssl
from lightning import Trainer
from rec_arena.models.graph_models import PyGLightGCN
from rec_arena.datasets import ML100K, RecDataModule
from rec_arena.configs.defaults.lightgcn import LightGCNConfig
from rec_arena.samplers import GraphRandomSampler
from rec_arena.metrics import MetricCalculator


def main():
    # Download data
    if not os.path.exists("../data/ml-100k/u.data"):
        os.makedirs("../data", exist_ok=True)
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "../data/ml-100k.zip",
        )
        with zipfile.ZipFile("../data/ml-100k.zip", "r") as z:
            z.extractall("../data/")
        os.remove("../data/ml-100k.zip")
    
    # Load dataset
    print("Loading MovieLens 100K dataset...")
    dataset = ML100K("../data/ml-100k/")
    dataset.load_data()
    print(f"Dataset loaded: {dataset.num_users} users, {dataset.num_items} items")
    
    # Create negative sampler for BPR loss (using GraphRandomSampler for 0-indexed items)
    negative_sampler = GraphRandomSampler(num_items=dataset.num_items, num_negatives=1, seed=42)
    
    # Create datamodule for graph format
    datamodule = RecDataModule(
        dataset,
        format="graph",
        batch_size=1024,
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
    
    # Check edge counts
    train_df, _, _ = dataset.split()
    original_interactions = len(train_df)
    total_edges = edge_index.shape[1]
    print(f"\nEdge count analysis:")
    print(f"Original training interactions: {original_interactions}")
    print(f"Total edges in edge_index: {total_edges}")
    print(f"Ratio (should be 2.0 for bidirectional): {total_edges / original_interactions:.1f}")
    print(f"Edge index shape: {edge_index.shape}")
    
    print("\n=== Training PyG LightGCN ===")
    
    # Train
    trainer = Trainer(
        max_epochs=100,
        accelerator="cpu",
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    trainer.fit(model, datamodule)
    
    # Check if parameters changed
    final_emb = model.pyg_model.embedding.weight.data
    emb_change = (final_emb - initial_emb).norm()
    print(f"\nParameter changes:")
    print(f"Embedding change: {emb_change:.6f}")
    print(f"Final embedding norm: {final_emb.norm():.4f}")
    
    # Evaluate on test set
    print("\n=== Evaluating PyG LightGCN ===")
    
    # Collect predictions and targets from test_step
    model.eval()
    test_loader = datamodule.test_dataloader()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            result = model.test_step(batch, 0)
            all_predictions.append(result["predictions"])
            all_targets.append(result["targets"])
    
    # Concatenate and calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    calculator = MetricCalculator(k_values=[10])
    results = calculator.calculate_all(all_predictions, all_targets)
    
    print(f"NDCG@10: {results['ndcg@10']:.4f}")
    print(f"Hit Rate@10: {results['hit_rate@10']:.4f}")
    print(f"Precision@10: {results['precision@10']:.4f}")
    print(f"Recall@10: {results['recall@10']:.4f}")
    
    # Generate sample recommendations
    print("\n=== Sample Recommendations ===")
    user_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    recommendations, scores = model.recommend(user_ids, k=10)
    print(f"Top-10 recommendations for users {user_ids.tolist()}:")
    for i, user_id in enumerate(user_ids):
        print(f"User {user_id}: {recommendations[i].tolist()}")


if __name__ == "__main__":
    main()
