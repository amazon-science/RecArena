"""Minimal example demonstrating LLaDA4Rec - Diffusion-based Sequential Recommendation."""

from lightning import Trainer
from rec_arena.models import LLaDA4Rec
from rec_arena.datasets import S3Dataset, RecDataModule
from rec_arena.configs.defaults import LLaDA4RecConfig
from rec_arena.losses.sequential import LLaDALoss


def main():
    # Load dataset
    dataset = S3Dataset(dataset_name="ml_1m", split_type="leave_one_out")
    dataset.load_data()

    # Create datamodule
    datamodule = RecDataModule(
        dataset,
        format="sequential",
        model_type="bert4rec",  # Use BERT4Rec format (bidirectional)
        batch_size=128,
        num_workers=0,
        num_negatives=0,  # LLaDA doesn't use negative sampling
        max_seq_length=100
    )
    datamodule.setup("fit")

    # Configure LLaDA4Rec
    config = LLaDA4RecConfig(
        vocab_size=dataset.num_items + 1,
        max_seq_length=100,
        embedding_dim=128,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        # Diffusion-specific parameters
        eps=0.01,  # Minimum masking ratio
        diffusion_steps=50,  # Iterative unmasking steps
        remasking_strategy="low_confidence",  # or "random"
        temperature=0.0,  # Greedy decoding
    )

    # Create model with LLaDA loss
    model = LLaDA4Rec(config)
    model.set_loss_fn(LLaDALoss())

    # Train
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(model, datamodule)

    # Test
    datamodule.setup("test")
    test_results = trainer.test(model, datamodule)
    
    print("\n" + "="*70)
    print("LLaDA4Rec Test Results:")
    print("="*70)
    for key, value in test_results[0].items():
        print(f"{key}: {value:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
