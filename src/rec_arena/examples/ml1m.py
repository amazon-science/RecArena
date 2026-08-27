"""Comprehensive ML-1M Test: All models and loss functions (1 epoch each)."""

from lightning import Trainer
from rec_arena.models import (
    SASRec,
    BERT4Rec,
    GRU4Rec,
    RecM,
    Caser,
    FMLPRec,
    HSTU,
    MLP4Rec,
    SLIM,
    ItemKNN,
    EASE,
)
from rec_arena.datasets import RecDataModule, TraditionalDataModule
from rec_arena.datasets.ensemble_datamodule import EnsembleRecDataModule
from rec_arena.configs.defaults.sasrec import SASRecConfig
from rec_arena.configs.defaults.bert4rec import BERT4RecConfig
from rec_arena.configs.defaults.gru4rec import GRU4RecConfig
from rec_arena.configs.defaults.recm import RecMConfig
from rec_arena.configs.defaults.caser import CaserConfig
from rec_arena.configs.defaults.fmlprec import FMLPRecConfig
from rec_arena.configs.defaults.hstu import HSTUConfig
from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
from rec_arena.configs.defaults.slim import SLIMConfig
from rec_arena.configs.defaults.itemknn import ItemKNNConfig
from rec_arena.configs.defaults.ease import EASEConfig
from rec_arena.datasets import S3Dataset
import os
import time

os.environ["AWS_PROFILE"] = "example-account"

# Load ML-1M dataset from S3
print("Loading MovieLens 1M dataset from S3...")
dataset = S3Dataset(dataset_name="ml_1m", split_type="leave_one_out")
dataset.load_data()
print(f"Dataset loaded: {dataset.num_users} users, {dataset.num_items} items\n")

# Test configurations
EPOCHS = 10
LOSS_FUNCTIONS = ["cross_entropy"]

# Sequential models to test
SEQUENTIAL_MODELS = [
    ("HSTU", HSTU, HSTUConfig, {"num_layers": 4}),
    ("GRU4Rec", GRU4Rec, GRU4RecConfig, {"num_layers": 2, "hidden_size": 128}),
    ("Caser", Caser, CaserConfig, {}),
    ("FMLPRec", FMLPRec, FMLPRecConfig, {"num_blocks": 2}),
    ("MLP4Rec", MLP4Rec, MLP4RecConfig, {"hidden_dims": [512, 256], "pooling": "mean"}),
    ("BERT4Rec", BERT4Rec, BERT4RecConfig, {"num_layers": 2, "num_heads": 2}),
    ("SASRec", SASRec, SASRecConfig, {"num_layers": 2, "num_heads": 2}),
]


# Traditional models to test
TRADITIONAL_MODELS = [
    ("SLIM", SLIM, SLIMConfig),
    ("ItemKNN", ItemKNN, ItemKNNConfig),
    ("EASE", EASE, EASEConfig),
]

print(f"{'='*70}")
print(f"COMPREHENSIVE MODEL TEST - {EPOCHS} EPOCHS EACH")
print(f"{'='*70}\n")

results = []


# Test Sequential Models
print("\n" + "=" * 70)
print("SEQUENTIAL MODELS")
print("=" * 70)

# Test Sequential Models with different loss functions
for model_name, model_class, config_class, extra_config in SEQUENTIAL_MODELS:
    for loss_type in LOSS_FUNCTIONS:
        start = time.time()
        print(f"\n{'='*70}")
        print(f"Testing: {model_name} with {loss_type.upper()}")
        print(f"{'='*70}")

        # Common config for sequential models
        config_dict = {
            "vocab_size": dataset.num_items + 1,
            "max_seq_length": 100,
            "embedding_dim": 128,
            "loss_type": loss_type,
            "compute_val_metrics": True,
            "val_k_values": [10],
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "early_stopping": False,
            **extra_config,
        }

        # Caser needs vertical_filter_size to match actual sequence length
        if model_name == "Caser":
            config_dict["vertical_filter_size"] = 100

        # BPR, BCE, and Sampled Softmax require negative samples
        num_negatives = 50 if loss_type in ["bpr", "bce", "sampled_softmax"] else 0

        # Create datamodule
        datamodule = RecDataModule(
            dataset,
            format="sequential",
            model_type=model_name.lower(),
            batch_size=128,
            num_workers=0,
            num_negatives=num_negatives,
            max_seq_length=100,
        )
        datamodule.setup("fit")

        # Create config and model
        config = config_class(**config_dict)
        model = model_class(config)

        # Train
        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="mps",
            devices="auto",
            enable_checkpointing=False,
            logger=False,
        )

        trainer.fit(model, datamodule)

        # Test
        datamodule.setup("test")
        test_results = trainer.test(model, datamodule, verbose=False)
        end = time.time()
        elapsed = end - start
        print(f"Time taken: {elapsed:.2f} seconds")

        if test_results and len(test_results) > 0:
            test_acc = test_results[0].get("test_acc@10", 0.0)
            test_ndcg = test_results[0].get("test_ndcg@10", 0.0)
            print(
                f"✓ {model_name} + {loss_type}: Acc@10={test_acc:.4f}, NDCG@10={test_ndcg:.4f}"
            )
            results.append((model_name, loss_type, test_acc, test_ndcg, elapsed))
        else:
            print(f"✗ {model_name} + {loss_type}: No test results")
            results.append((model_name, loss_type, 0.0, 0.0, elapsed))


# Test Traditional Models first
print("\n" + "=" * 70)
print("TRADITIONAL MODELS")
print("=" * 70)

for model_name, model_class, config_class in TRADITIONAL_MODELS:
    start = time.time()
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    try:
        import scipy.sparse as sp
        import numpy as np

        train_df = dataset.train_df
        test_df = dataset.test_df

        # Calculate dimensions from actual data
        num_users = max(train_df["user_id"].max(), test_df["user_id"].max()) + 1
        num_items = max(train_df["item_id"].max(), test_df["item_id"].max()) + 1

        # Create sparse matrix with 0-indexed items
        user_ids = train_df["user_id"].values
        item_ids = (
            train_df["item_id"].values - 1
        )  # Convert to 0-indexed for matrix columns
        data = np.ones(len(train_df))
        train_matrix = sp.csr_matrix(
            (data, (user_ids, item_ids)),
            shape=(num_users, num_items),
            dtype=np.float32,
        )

        # Special config for EASE: use ensemble with 20 models
        if model_name == "EASE":
            # Logarithmically spaced lambdas from 50 to 5000
            import numpy as np

            ensemble_lambdas = np.logspace(np.log10(50), np.log10(5000), 20).tolist()
            config = config_class(
                num_users=num_users,
                num_items=num_items,
                ensemble_lambdas=ensemble_lambdas,
                ensemble_method="mean",
            )
        else:
            config = config_class(num_users=num_users, num_items=num_items)
        model = model_class(config)

        print(f"Fitting {model_name}...")
        model.fit(train_matrix)
        print(f"✓ Training completed")

        # Evaluate
        test_users = test_df["user_id"].values
        test_items = test_df["item_id"].values
        unique_test_users = np.unique(test_users)

        user_vectors = train_matrix[unique_test_users]
        if model_name == "EASE":
            all_scores = user_vectors.toarray() @ model.B
        elif model_name == "ItemKNN":
            all_scores = user_vectors.toarray() @ model.similarity_matrix
        elif model_name == "SLIM":
            all_scores = (user_vectors @ model.W).toarray()

        all_scores[train_matrix[unique_test_users].toarray() > 0] = -np.inf

        import torch

        predictions = torch.from_numpy(all_scores).float()

        targets = torch.zeros(len(unique_test_users), dtype=torch.long)
        for i, user_id in enumerate(unique_test_users):
            user_test_items = test_items[test_users == user_id]
            targets[i] = (
                user_test_items[0] - 1
            )  # Convert to 0-indexed for predictions matrix

        from rec_arena.metrics import MetricCalculator

        metric_calc = MetricCalculator(k_values=[10])
        metrics = metric_calc.calculate_all(predictions, targets)

        elapsed = time.time() - start
        test_recall = metrics.get("recall@10", 0.0)
        test_ndcg = metrics.get("ndcg@10", 0.0)
        print(f"✓ Results: Recall@10={test_recall:.4f}, NDCG@10={test_ndcg:.4f}")
        print(f"Time taken: {elapsed:.2f} seconds")
        results.append((model_name, "traditional", test_recall, test_ndcg, elapsed))

    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Error: {str(e)}")
        results.append((model_name, "traditional", 0.0, 0.0, elapsed))

# Test Ensemble Model
print("\n" + "=" * 70)
print("ENSEMBLE MODEL")
print("=" * 70)

# Test RecM (ensemble model)
start = time.time()
print(f"\n{'='*70}")
print(f"Testing: RecM (Ensemble)")
print(f"{'='*70}")

datamodule = EnsembleRecDataModule(
    dataset,
    ensemble_samplers=["random"] * 4,
    ensemble_num_negatives=[0] * 4,
    ensemble_size=4,
    format="sequential",
    model_type="recm",
    batch_size=128,
    num_workers=0,
    max_seq_length=100,
)
datamodule.setup("fit")

config = RecMConfig(
    vocab_size=dataset.num_items + 1,
    max_seq_length=100,
    embedding_dim=128,
    num_layers=2,
    ensemble_size=4,
    ensemble_loss_functions=["cross_entropy"] * 4,
    compute_val_metrics=True,
    val_k_values=[10],
    lr=1e-3,
    early_stopping=False,
)

model = RecM(config)

trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="mps",
    devices="auto",
    enable_checkpointing=False,
    logger=False,
)

trainer.fit(model, datamodule)
datamodule.setup("test")
test_results = trainer.test(model, datamodule, verbose=False)
end = time.time()
elapsed = end - start
print(f"Time taken: {elapsed:.2f} seconds")

# Extract RecM results immediately
if test_results and len(test_results) > 0:
    test_acc = test_results[0].get("test_acc@10", 0.0)
    test_ndcg = test_results[0].get("test_ndcg@10", 0.0)
    print(f"✓ RecM: Acc@10={test_acc:.4f}, NDCG@10={test_ndcg:.4f}")
    results.append(("RecM", "ensemble", test_acc, test_ndcg, elapsed))
else:
    print(f"✗ RecM: No test results")
    results.append(("RecM", "ensemble", 0.0, 0.0, elapsed))


# Print summary
print(f"\n{'='*70}")
print("SUMMARY OF RESULTS")
print(f"{'='*70}")
print(f"{'Model':<15} {'Loss':<20} {'Metric@10':>10} {'NDCG@10':>10} {'Time (s)':>10}")
print(f"{'-'*70}")
for model, loss, metric, ndcg, elapsed in results:
    print(f"{model:<15} {loss:<20} {metric:>10.4f} {ndcg:>10.4f} {elapsed:>10.2f}")

total_time = sum(r[4] for r in results)
print(f"{'-'*70}")
print(f"{'TOTAL':<15} {'':<20} {'':<10} {'':<10} {total_time:>10.2f}")

print(f"\n{'='*70}")
print(f"🎉 All tests completed! Total: {total_time:.2f}s ({total_time/60:.2f}min)")
print(f"{'='*70}")
