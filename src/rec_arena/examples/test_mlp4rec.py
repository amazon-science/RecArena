"""Test MLP4Rec on ML-1M dataset."""

from lightning import Trainer
from rec_arena.models import MLP4Rec
from rec_arena.datasets import RecDataModule, S3Dataset
from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
import os
import time

os.environ["AWS_PROFILE"] = "example-account"

# Load dataset
print("Loading MovieLens 1M dataset...")
dataset = S3Dataset(dataset_name="ml_1m", split_type="leave_one_out")
dataset.load_data()
print(f"Dataset: {dataset.num_users} users, {dataset.num_items} items\n")

# Test MLP4Rec
start = time.time()
print("Testing MLP4Rec with cross_entropy loss")
print("=" * 70)

# Create datamodule
datamodule = RecDataModule(
    dataset,
    format="sequential",
    model_type="mlp4rec",
    batch_size=128,
    num_workers=0,
    num_negatives=0,
    max_seq_length=100,
)
datamodule.setup("fit")

# Create config
config = MLP4RecConfig(
    vocab_size=dataset.num_items + 1,
    max_seq_length=100,
    embedding_dim=128,
    hidden_dims=[512, 512],
    pooling="last",
    loss_type="cross_entropy",
    compute_val_metrics=True,
    val_k_values=[10],
    lr=1e-03,
    weight_decay=1e-5,
)

# Create model
model = MLP4Rec(config)

# Train
trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    devices="auto",
    enable_checkpointing=False,
    logger=False,
)

trainer.fit(model, datamodule)

# Test
datamodule.setup("test")
test_results = trainer.test(model, datamodule, verbose=False)
elapsed = time.time() - start

if test_results:
    test_acc = test_results[0].get("test_acc@10", 0.0)
    test_ndcg = test_results[0].get("test_ndcg@10", 0.0)
    print(f"\n✓ MLP4Rec Results:")
    print(f"  Recall@10: {test_acc:.4f}")
    print(f"  NDCG@10: {test_ndcg:.4f}")
    print(f"  Time: {elapsed:.2f}s")
else:
    print("✗ No test results")
