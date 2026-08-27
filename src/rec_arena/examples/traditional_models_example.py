"""Example: Training traditional models (EASE, SLIM, ItemKNN) on S3 data."""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import os
from rec_arena.models import EASE, SLIM, ItemKNN
from rec_arena.configs.defaults.ease import EASEConfig
from rec_arena.configs.defaults.slim import SLIMConfig
from rec_arena.configs.defaults.itemknn import ItemKNNConfig
from rec_arena.metrics import MetricCalculator

# Set AWS profile for AWS authentication
os.environ["AWS_PROFILE"] = "example-account"

bucket_name = "example-bucket"

# Load train/test splits (LOO: leave-one-out)
print("Loading data from S3...")
train_df = pd.read_parquet(
    f"s3://{bucket_name}/recarena/ml_1m/leave_one_out/train.parquet"
)
test_df = pd.read_parquet(
    f"s3://{bucket_name}/recarena/ml_1m/leave_one_out/test.parquet"
)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Get dataset dimensions
num_users = max(train_df["user_id"].max(), test_df["user_id"].max()) + 1
num_items = max(train_df["item_id"].max(), test_df["item_id"].max()) + 1

print(f"\nDataset: {num_users} users, {num_items} items")


# Create sparse interaction matrix from train data
def create_sparse_matrix(df, num_users, num_items):
    """Create sparse user-item interaction matrix."""
    user_ids = df["user_id"].values
    item_ids = df["item_id"].values
    data = np.ones(len(df))

    matrix = sp.csr_matrix(
        (data, (user_ids, item_ids)), shape=(num_users, num_items), dtype=np.float32
    )
    return matrix


train_matrix = create_sparse_matrix(train_df, num_users, num_items)
test_matrix = create_sparse_matrix(test_df, num_users, num_items)

print(
    f"Train matrix: {train_matrix.shape}, density: {train_matrix.nnz / (num_users * num_items):.4f}"
)

# Initialize metric calculator
metric_calc = MetricCalculator(k_values=[10])

# ============================================================================
# 1. EASE
# ============================================================================
print("\n" + "=" * 60)
print("Training EASE (Embarrassingly Shallow Autoencoders)")
print("=" * 60)

config = EASEConfig(num_users=num_users, num_items=num_items, reg_lambda=500.0)

model = EASE(config)
print("Fitting EASE (closed-form solution)...")
model.fit(train_matrix)
print("✅ EASE training completed!")

# Evaluate on test set
print("\nEvaluating EASE...")
test_users = test_df["user_id"].values
test_items = test_df["item_id"].values

# Get predictions for test users (full score distribution)
unique_test_users = np.unique(test_users)
user_vectors = train_matrix[unique_test_users].toarray()
all_scores = user_vectors @ model.B  # Full scores for all items

# Mask out training items (set to -inf)
all_scores[train_matrix[unique_test_users].toarray() > 0] = -np.inf

predictions = torch.from_numpy(all_scores).float()

# Create targets tensor
targets = torch.zeros(len(unique_test_users), dtype=torch.long)
for i, user_id in enumerate(unique_test_users):
    user_test_items = test_items[test_users == user_id]
    targets[i] = user_test_items[0]  # LOO: one item per user

# Calculate metrics
metrics = metric_calc.calculate_all(predictions, targets)
print(
    f"EASE - Hit Rate@10: {metrics['hit_rate@10']:.4f}, NDCG@10: {metrics['ndcg@10']:.4f}"
)

# ============================================================================
# 2. ItemKNN
# ============================================================================
print("\n" + "=" * 60)
print("Training ItemKNN (Item-based K-Nearest Neighbors)")
print("=" * 60)

config = ItemKNNConfig(
    num_users=num_users,
    num_items=num_items,
    k=100,
    similarity="cosine",
    shrinkage=100.0,
)

model = ItemKNN(config)
print("Fitting ItemKNN (computing similarities)...")
model.fit(train_matrix)
print("✅ ItemKNN training completed!")

# Evaluate (full score distribution)
user_vectors = train_matrix[unique_test_users].toarray()
all_scores = user_vectors @ model.similarity_matrix

# Mask out training items (set to -inf)
all_scores[train_matrix[unique_test_users].toarray() > 0] = -np.inf

predictions = torch.from_numpy(all_scores).float()
metrics = metric_calc.calculate_all(predictions, targets)
print(
    f"ItemKNN - Hit Rate@10: {metrics['hit_rate@10']:.4f}, NDCG@10: {metrics['ndcg@10']:.4f}"
)

# ============================================================================
# 3. SLIM
# ============================================================================
print("\n" + "=" * 60)
print("Training SLIM (Sparse Linear Methods)")
print("=" * 60)
print("Note: SLIM training may take several minutes...")

config = SLIMConfig(num_users=num_users, num_items=num_items, alpha=0.1, l1_ratio=0.1)

model = SLIM(config)
print("Fitting SLIM (elastic net per item)...")
model.fit(train_matrix)
print("✅ SLIM training completed!")

# Evaluate (full score distribution)
user_vectors = train_matrix[unique_test_users]
all_scores = (user_vectors @ model.W).toarray()

# Mask out training items (set to -inf)
all_scores[train_matrix[unique_test_users].toarray() > 0] = -np.inf

predictions = torch.from_numpy(all_scores).float()
metrics = metric_calc.calculate_all(predictions, targets)
print(
    f"SLIM - Hit Rate@10: {metrics['hit_rate@10']:.4f}, NDCG@10: {metrics['ndcg@10']:.4f}"
)

print("\n" + "=" * 60)
print("🎉 All traditional models trained and evaluated!")
print("=" * 60)
