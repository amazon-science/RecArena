"""SASRec vs HSTU comparison: 100 epochs with early stopping and LR decay."""

from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from rec_arena.models import SASRec, BERT4Rec, HSTU, MLP4Rec, FuXi, FuXiGamma
from rec_arena.datasets import RecDataModule
from rec_arena.configs.defaults.sasrec import SASRecConfig
from rec_arena.configs.defaults.bert4rec import BERT4RecConfig
from rec_arena.configs.defaults.hstu import HSTUConfig
from rec_arena.configs.defaults.mlp4rec import MLP4RecConfig
from rec_arena.configs.defaults.fuxi import FuXiConfig
from rec_arena.configs.defaults.fuxi_gamma import FuXiGammaConfig
from rec_arena.datasets import S3Dataset
import os
import time
import json
import torch


class NDCGEarlyStopping(Callback):
    """Early stopping based on NDCG."""

    def __init__(self, patience=200, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.wait_count = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if "val_ndcg@10" not in logs:
            return

        current = logs["val_ndcg@10"].item()

        if self.best_score is None:
            self.best_score = current
        elif (self.mode == "max" and current > self.best_score) or (
            self.mode == "min" and current < self.best_score
        ):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                trainer.should_stop = True


os.environ["AWS_PROFILE"] = "example-account"

# Results file
RESULTS_FILE = "model_comparison_results.json"

# Load ML-1M dataset from S3
print("Loading MovieLens 1M dataset from S3...")
dataset = S3Dataset(dataset_name="ml_1m", split_type="leave_one_out")
dataset.load_data()
print(f"Dataset loaded: {dataset.num_users} users, {dataset.num_items} items\n")

EPOCHS = 1000
LOSS_FUNCTIONS = [
    "cross_entropy",
    "bce",
    "sampled_softmax",
]

# Sequential models to test
SEQUENTIAL_MODELS = [
    # ("BERT4Rec", BERT4Rec, BERT4RecConfig, {"num_layers": 4, "num_heads": 2, "embedding_dim": 64}),
    # ("HSTU", HSTU, HSTUConfig, {}),  # Use default HSTU-Large config (50-dim, 8 layers)
    # (
    #    "FuXi",
    #    FuXi,
    #    FuXiConfig,
    #    {
    #        "num_layers": 8,
    #        "num_heads": 2,
    #        "embedding_dim": 50,
    #        "linear_dim": 50,
    #        "attention_dim": 50,
    #        "ffn_multiply": 1.0,
    #        "ffn_single_stage": False,
    #        "dropout_rate": 0.2,
    #    },
    # ),
    (
        "FuXiGamma",
        FuXiGamma,
        FuXiGammaConfig,
        {
            "num_layers": 8,
            "num_heads": 2,
            "embedding_dim": 50,
            "linear_dim": 25,
            "attention_dim": 50,
            "ffn_multiply": 1.0,
            "dropout_rate": 0.2,
        },
    ),
    (
        "SASRec",
        SASRec,
        SASRecConfig,
        {
            "num_layers": 4,
            "num_heads": 2,
            "use_ligr": True,
            "position_config": {"type": "rope", "base": 1000},
            "embedding_dim": 64,
        },
    ),
    (
        "MLP4Rec",
        MLP4Rec,
        MLP4RecConfig,
        {"hidden_dims": [512, 512], "pooling": "multi", "embedding_dim": 64},
    ),
]

print(f"{'='*70}")
print(f"MODEL COMPARISON - {EPOCHS} EPOCHS WITH EARLY STOPPING & LR DECAY")
print(f"{'='*70}\n")

results = []

for model_name, model_class, config_class, extra_config in SEQUENTIAL_MODELS:
    for loss_type in LOSS_FUNCTIONS:
        start = time.time()
        print(f"\n{'='*70}")
        print(f"Testing: {model_name} with {loss_type.upper()}")
        print(f"{'='*70}")

        # Set num_negatives based on loss type (use paper settings)
        if loss_type == "sampled_softmax":
            num_negatives = 128  # HSTU paper uses 128
        elif loss_type in ["bpr", "bce", "gbce"]:
            num_negatives = 50
        else:
            num_negatives = 0

        # Create datamodule
        datamodule = RecDataModule(
            dataset,
            format="sequential",
            model_type=model_name.lower(),
            batch_size=128,
            num_workers=0,
            num_negatives=num_negatives,
            max_seq_length=200,
        )
        datamodule.setup("fit")

        # Create config with LR scheduler
        config = config_class(
            vocab_size=dataset.num_items
            + 3,  # Vocab: [0=PAD, 1=UNK, 2=MASK, 3...num_items+2=items]
            max_seq_length=200,
            loss_type=loss_type,  # Use the loss_type from loop
            compute_val_metrics=True,
            val_k_values=[10],
            metric_compute_interval=1,  # Compute NDCG every epoch
            lr=1e-3,
            weight_decay=0,  # All paper configs use 0
            **extra_config,
        )
        # Add loss kwargs for sampled losses (paper settings)
        if loss_type == "sampled_softmax":
            config.loss_kwargs = {"temperature": 0.05, "l2_norm": True}
        # Add LR scheduler
        config.scheduler = {
            "type": "reduce_on_plateau",
            "patience": 15,
            "factor": 0.8,
            "monitor": "val_acc@10",
            "mode": "max",
        }

        # Create model
        model = model_class(config)

        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {trainable_params:,}")

        # Early stopping on NDCG with 200 epoch patience
        early_stop = NDCGEarlyStopping(patience=200, mode="max")

        # Checkpoint to save best model based on val_ndcg@10
        checkpoint = ModelCheckpoint(
            monitor="val_ndcg@10",
            mode="max",
            save_top_k=1,
            dirpath="checkpoints",
            filename=f"{model_name}_{loss_type}_best",
        )

        # Train
        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            enable_checkpointing=True,
            logger=False,
            callbacks=[early_stop, checkpoint],
            precision="bf16-mixed",  # BF16 mixed precision for A10G (faster + less memory)
        )

        trainer.fit(model, datamodule)
        print(f"Stopped at epoch {trainer.current_epoch + 1}")

        # Load best checkpoint - manually load state dict to avoid PyTorch 2.6 pickle issue
        if checkpoint.best_model_path:
            print(f"Loading best model from {checkpoint.best_model_path}")
            import torch

            ckpt = torch.load(checkpoint.best_model_path, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])

        # Test
        datamodule.setup("test")
        test_results = trainer.test(model, datamodule, verbose=False)
        elapsed = time.time() - start

        if test_results:
            test_acc = test_results[0].get("test_acc@10", 0.0)
            test_ndcg = test_results[0].get("test_ndcg@10", 0.0)
            print(f"✓ Recall@10: {test_acc:.4f}, NDCG@10: {test_ndcg:.4f}")
            print(f"  Time: {elapsed:.2f}s")
            results.append(
                (
                    f"{model_name}_{loss_type}",
                    test_acc,
                    test_ndcg,
                    trainable_params,
                    elapsed,
                )
            )
        else:
            print("✗ No test results")
            results.append(
                (f"{model_name}_{loss_type}", 0.0, 0.0, trainable_params, elapsed)
            )

        # Save intermediate results after each model
        with open(RESULTS_FILE, "w") as f:
            json.dump(
                {
                    "results": [
                        {
                            "model": r[0],
                            "recall@10": r[1],
                            "ndcg@10": r[2],
                            "params": r[3],
                            "time": r[4],
                        }
                        for r in results
                    ]
                },
                f,
                indent=2,
            )
        print(f"Saved intermediate results to {RESULTS_FILE}")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(
    f"{'Model_Loss':<25} {'Recall@10':>12} {'NDCG@10':>12} {'Params':>12} {'Time(s)':>10}"
)
print(f"{'':25} {'':12} {'':12} {'':12} {'(early stop)':>10}")
print("-" * 70)
for name, recall, ndcg, params, time_taken in results:
    print(f"{name:<25} {recall:>12.4f} {ndcg:>12.4f} {params:>12,} {time_taken:>10.2f}")
print("=" * 70)

# Save final results
with open(RESULTS_FILE, "w") as f:
    json.dump(
        {
            "results": [
                {
                    "model": r[0],
                    "recall@10": r[1],
                    "ndcg@10": r[2],
                    "params": r[3],
                    "time": r[4],
                }
                for r in results
            ]
        },
        f,
        indent=2,
    )
print(f"\nFinal results saved to {RESULTS_FILE}")
