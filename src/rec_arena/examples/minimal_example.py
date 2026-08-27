"""Comprehensive example: Testing all models with all compatible loss functions."""

from lightning import Trainer
from rec_arena.models import SASRec, BERT4Rec, GRU4Rec, NCF, TwoTower, RecM, BPRMF, Caser, SimpleX, FMLPRec, PyGLightGCN
from rec_arena.datasets import ML100K, RecDataModule
from rec_arena.datasets.ensemble_datamodule import EnsembleRecDataModule
from rec_arena.configs.defaults.sasrec import SASRecConfig
from rec_arena.configs.defaults.bert4rec import BERT4RecConfig
from rec_arena.configs.defaults.gru4rec import GRU4RecConfig
from rec_arena.configs.defaults.ncf import NCFConfig
from rec_arena.configs.defaults.twotower import TwoTowerConfig
from rec_arena.configs.defaults.recm import RecMConfig
from rec_arena.configs.defaults.bprmf import BPRMFConfig
from rec_arena.configs.defaults.caser import CaserConfig
from rec_arena.configs.defaults.simplex import SimpleXConfig
from rec_arena.configs.defaults.fmlprec import FMLPRecConfig
from rec_arena.configs.defaults.lightgcn import LightGCNConfig
from rec_arena.samplers import GraphRandomSampler
import torch
import os
import urllib.request
import zipfile
import ssl

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

# Model configurations
models_configs = [
    # Implicit models
    (
        "BPRMF",
        BPRMF,
        BPRMFConfig,
        "implicit",
        None,
        ["bpr"],
    ),
    (
        "SimpleX",
        SimpleX,
        SimpleXConfig,
        "implicit",
        None,
        ["bpr"],
    ),
    (
        "NCF",
        NCF,
        NCFConfig,
        "implicit",
        None,
        ["bce", "bpr"],
    ),
    (
        "TwoTower",
        TwoTower,
        TwoTowerConfig,
        "implicit",
        None,
        ["bce", "bpr"],
    ),
    # Sequential models
    (
        "Caser",
        Caser,
        CaserConfig,
        "sequential",
        "caser",
        ["cross_entropy", "bce", "bpr"],
    ),
    (
        "FMLPRec",
        FMLPRec,
        FMLPRecConfig,
        "sequential",
        "fmlprec",
        ["cross_entropy", "bce", "bpr"],
    ),
    (
        "RecM",
        RecM,
        RecMConfig,
        "sequential",
        "recm",
        ["ensemble_mixed"],  # Special case for mixed ensemble losses
    ),
    (
        "SASRec",
        SASRec,
        SASRecConfig,
        "sequential",
        "sasrec",
        ["cross_entropy", "bce", "sampled_softmax", "bpr", "gbce"],
    ),
    (
        "BERT4Rec",
        BERT4Rec,
        BERT4RecConfig,
        "sequential",
        "bert4rec",
        ["cross_entropy", "bce", "sampled_softmax", "bpr"],
    ),
    (
        "GRU4Rec",
        GRU4Rec,
        GRU4RecConfig,
        "sequential",
        "gru4rec",
        ["cross_entropy", "bce", "sampled_softmax", "bpr", "gbce"],
    ),
    # Graph models
    (
        "PyGLightGCN",
        PyGLightGCN,
        LightGCNConfig,
        "graph",
        None,
        ["bpr"],
    ),
]

print(f"\n=== Testing {len(models_configs)} models with multiple loss functions ===")

for (
    model_name,
    model_class,
    config_class,
    data_format,
    model_type,
    loss_types,
) in models_configs:
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    for loss_type in loss_types:
        print(f"\n--- {model_name} with {loss_type.upper()} loss ---")

        # Determine if negative sampling is needed
        needs_negatives = loss_type in ["bce", "sampled_softmax", "bpr", "gbce"]

        # Create datamodule
        if model_name == "RecM":
            # Use EnsembleRecDataModule for RecM
            datamodule = EnsembleRecDataModule(
                dataset,
                ensemble_samplers=4 * ["random", "popularity"],  # 2 different samplers
                ensemble_num_negatives=4 * [25, 75],  # 25 for random, 75 for popularity
                ensemble_size=8,  # 4 heads each: 0-3 use random+25, 4-7 use popularity+75
                format="sequential",
                model_type=model_type,
                batch_size=16,
                num_workers=0,
                max_seq_length=200,
            )
        elif data_format == "sequential":
            datamodule = RecDataModule(
                dataset,
                format="sequential",
                model_type=model_type,
                batch_size=16,
                num_workers=0,
                num_negatives=50 if needs_negatives else 0,
                max_seq_length=200,
            )
        elif data_format == "graph":
            graph_sampler = GraphRandomSampler(num_items=dataset.num_items, num_negatives=5, seed=42) if needs_negatives else None
            datamodule = RecDataModule(
                dataset,
                format="graph",
                batch_size=16,
                num_workers=0,
                num_negatives=5 if needs_negatives else 0,
                negative_sampler=graph_sampler,
            )
        else:  # implicit
            datamodule = RecDataModule(
                dataset,
                format="implicit",
                batch_size=16,
                num_workers=0,
                num_negatives=4 if needs_negatives else 0,
                max_seq_length=200,
            )

        # Create model config with validation metrics enabled
        # Implicit models: use num_users, num_items
        if model_name == "NCF":
            config = config_class(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                hidden_dims=[128, 32],
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )
        elif model_name == "BPRMF":
            config = config_class(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                embedding_dim=64,
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )
        elif model_name == "TwoTower":
            config = config_class(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                user_tower_dims=[128, 64],
                item_tower_dims=[128, 64],
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )
        elif model_name == "SimpleX":
            config = config_class(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                embedding_dim=64,
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )
        # Sequential models: use vocab_size, max_seq_length
        elif model_name == "GRU4Rec":
            config = config_class(
                vocab_size=dataset.num_items + 3,
                max_seq_length=200,
                embedding_dim=64,
                hidden_size=64,  # Match embedding_dim for GRU4Rec
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )
        elif model_name == "RecM":
            config = config_class(
                vocab_size=dataset.num_items + 3,
                max_seq_length=200,
                embedding_dim=64,
                num_layers=2,
                ensemble_size=8,
                ensemble_loss_functions=[
                    "cross_entropy",  # No negatives needed
                    "bpr",  # Needs negatives
                    "bce",  # Needs negatives
                    "sampled_softmax",  # Needs negatives
                ],  # 2 heads each
                compute_val_metrics=True,
                val_k_values=[10],
            )
        elif model_name in ["Caser", "FMLPRec"]:
            config = config_class(
                vocab_size=dataset.num_items + 3,
                max_seq_length=200,
                embedding_dim=128,
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )
        elif model_name == "PyGLightGCN":
            config = config_class(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                embedding_dim=64,
                num_layers=3,
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )
        else:
            config = config_class(
                vocab_size=dataset.num_items + 3,
                max_seq_length=200,
                embedding_dim=128,
                num_layers=4,
                loss_type=loss_type,
                compute_val_metrics=True,
                val_k_values=[10],
            )

        # Create model
        model = model_class(config)

        # Set graph data for graph models
        if data_format == "graph":
            datamodule.setup()
            model.set_graph_data(datamodule.get_edge_index())

        try:

            # Regular Lightning training for neural models
            trainer_kwargs = {
                "max_epochs": 2,
                "accelerator": "cpu",
                "enable_progress_bar": True,
                "logger": False,
                "enable_checkpointing": False,
                "enable_model_summary": True,
            }
            
            # Add gradient clipping for graph models
            if model_name == "PyGLightGCN":
                trainer_kwargs["gradient_clip_val"] = 1.0
                trainer_kwargs["gradient_clip_algorithm"] = "norm"
            
            trainer = Trainer(**trainer_kwargs)

            trainer.fit(model, datamodule)

            # Evaluate on test set using Lightning's test method
            test_results = trainer.test(model, datamodule, verbose=False)

            from rec_arena.metrics import MetricCalculator

            calculator = MetricCalculator(k_values=[10])

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

            results = calculator.calculate_all(all_predictions, all_targets)
            ndcg_10 = results["ndcg@10"]
            hit_rate_10 = results["hit_rate@10"]

            print(
                f"✅ {model_name} + {loss_type.upper()} - SUCCESS | NDCG@10: {ndcg_10:.4f} | Hit Rate@10: {hit_rate_10:.4f}"
            )

        except Exception as e:
            print(f"❌ {model_name} + {loss_type.upper()} - FAILED: {str(e)}")
            import traceback

            traceback.print_exc()

            # Exit on failure
            print("\n🛑 Stopping execution due to failure")
            exit(1)

print(f"\n{'='*60}")
print("🎉 All model + loss function combinations tested!")
print(f"{'='*60}")
