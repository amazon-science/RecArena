"""Robustness experiments: model size and depth ablation."""
import argparse, os, time, tempfile
import numpy as np, pandas as pd, filelock
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from rec_arena.configs.defaults.sasrec import SASRecConfig
from rec_arena.datasets import S3Dataset, RecDataModule
from rec_arena.losses import get_loss_function
from rec_arena.models import SASRec

CONFIGS = {
    "baseline": {"position_config": {"type": "learnable"}, "use_ligr": False, "use_rms_norm": False},
    "rope+ligr": {"position_config": {"type": "rope", "base": 10000}, "use_ligr": True, "use_rms_norm": False},
    "rope+ligr+rms": {"position_config": {"type": "rope", "base": 10000}, "use_ligr": True, "use_rms_norm": True},
}
DATASETS = ["ml_100k", "ml_1m", "ratebeer", "goodreads"]

def run_exp(dataset_name, arch_config, arch_name, embedding_dim=64, num_layers=2, max_epochs=500):
    dataset = S3Dataset(dataset_name=dataset_name, split_type="leave_one_out",
                        s3_bucket=os.environ.get("RECARENA_S3_BUCKET"))
    dataset.load_data()
    max_seq_length = min(int(np.percentile(dataset.train_df.groupby("user_id").size().values, 75)), 200)
    
    dm = RecDataModule(dataset, format="sequential", model_type="sasrec", batch_size=128,
                       num_workers=0, num_negatives=0, max_seq_length=max_seq_length)
    dm.setup("fit")
    
    config = SASRecConfig(vocab_size=dataset.num_items+3, max_seq_length=max_seq_length,
                          embedding_dim=embedding_dim, num_heads=2, num_layers=num_layers,
                          loss_type="cross_entropy", lr=1e-3, weight_decay=1e-6,
                          metric_compute_interval=10, **arch_config)
    model = SASRec(config)
    model.set_loss_fn(get_loss_function("cross_entropy", "sequential"))
    
    ckpt_dir = tempfile.mkdtemp()
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor="val_ndcg@10",
                              mode="max", save_top_k=1, every_n_epochs=10)
    
    num_train = len(dataset.train_df)
    limit_batches = min(200, num_train // 128) if num_train // 128 > 200 else None
    
    trainer = Trainer(max_epochs=max_epochs, accelerator="cuda", precision="16-mixed",
                      enable_checkpointing=True, logger=False, enable_progress_bar=True,
                      limit_train_batches=limit_batches,
                      callbacks=[EarlyStopping(monitor="val_ndcg@10", patience=100, mode="max"), ckpt_cb])
    
    start = time.time()
    try:
        trainer.fit(model, dm)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"skipped": True, "reason": f"OOM: {e}", "dataset": dataset_name, "arch": arch_name}
        raise
    train_time = time.time() - start
    
    dm.setup("test")
    best = ckpt_cb.best_model_path
    res = trainer.test(model, dm, ckpt_path=best if best else None, verbose=False, weights_only=False)
    
    return {
        "dataset": dataset_name, "arch": arch_name, "embedding_dim": embedding_dim,
        "num_layers": num_layers, "train_time_s": round(train_time, 2),
        "epochs_trained": trainer.current_epoch + 1,
        **{k: round(v, 4) for k, v in res[0].items()},
    }

def append_result(output_dir, name, result):
    csv_path = output_dir / f"{name}.csv"
    lock_path = output_dir / f"{name}.csv.lock"
    with filelock.FileLock(lock_path):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        else:
            df = pd.DataFrame([result])
        df.to_csv(csv_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["size", "depth"], required=True)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    experiments = []
    if args.experiment == "size":
        for ds in DATASETS:
            for d in [64, 128, 256, 512]:
                for name, cfg in CONFIGS.items():
                    experiments.append((ds, cfg, f"{name}_d{d}", d, 2))
    else:
        for ds in DATASETS:
            for L in [1, 2, 4]:
                for name, cfg in CONFIGS.items():
                    experiments.append((ds, cfg, f"{name}_L{L}", 64, L))
    
    for i, (ds, cfg, arch_name, d, L) in enumerate(experiments):
        print(f"\\n[{i+1}/{len(experiments)}] {ds} | {arch_name}")
        result = run_exp(ds, cfg, arch_name, embedding_dim=d, num_layers=L)
        append_result(output_dir, f"robustness_{args.experiment}", result)
        print(f"  NDCG@10={result.get('test_ndcg@10', 'skipped')}")
    
    print(f"\\nDone! Results in {output_dir}/robustness_{args.experiment}.csv")

if __name__ == "__main__":
    main()
