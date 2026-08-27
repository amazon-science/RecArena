import torch
import torch.nn as nn
import lightning as pl
from typing import Dict, Any, Optional, Tuple
from .base import BaseModel
from ..configs.validation import validate_config
from ..utils.logging import setup_logger
from ..utils.memory import clear_memory

logger = setup_logger(__name__)


class DeepModel(BaseModel, pl.LightningModule):
    """Base class for deep learning recommendation models using PyTorch Lightning."""

    # Implicit models score via module-forward lookups (sampled bce/bpr), so
    # their embedding tables are sparse-safe when opted in.
    SUPPORTS_SPARSE = True

    def __init__(self, config: Dict[str, Any]):
        # Validate configuration for implicit models
        validate_config(config, "implicit")

        BaseModel.__init__(self, config)
        pl.LightningModule.__init__(self)

        self.num_users = config.get("num_users")
        self.num_items = config.get("num_items")
        self.embedding_dim = config.get("embedding_dim", 64)
        self.lr = config.get("lr", 1e-3)
        self.gradient_clip_val = config.get("gradient_clip_val", 1.0)

        # Sparse embedding eligibility. Implicit models (NCF/TwoTower/SimpleX/
        # BPRMF) always train with sampled losses (bce/bpr) that score via
        # module-forward lookups, so they are sparse-safe when opted in.
        # Subclasses read `self._sparse_embeddings` when building their tables.
        from ..utils.sparse_optim import sparse_embeddings_eligible

        self._sparse_embeddings, _sparse_reason = sparse_embeddings_eligible(
            config, type(self)
        )
        if getattr(config, "sparse_embeddings", False) and not self._sparse_embeddings:
            import warnings

            warnings.warn(
                f"sparse_embeddings requested but disabled: {_sparse_reason}. "
                "Falling back to dense embedding tables.",
                stacklevel=2,
            )

        # Store embedding configs for subclasses
        self.user_embedding_config = config.get(
            "user_embedding_config", {"type": "standard"}
        )
        self.item_embedding_config = config.get(
            "item_embedding_config", {"type": "standard"}
        )

        # Optional metrics during validation
        self.compute_val_metrics = config.get("compute_val_metrics", False)
        self.val_k_values = config.get("val_k_values", [10])
        self._metric_calculator = None
        # Optional per-user TRAIN-history table [num_users, Lmax] of 3-indexed
        # item ids, populated by the harness (set_eval_history). Used to mask a
        # user's already-seen items during VALIDATION metric computation so the
        # val NDCG that drives early stopping / best-checkpoint selection matches
        # the masked full-sort protocol used at TEST time (and RecBole's masked
        # valid metric). Without this the monitored signal is deflated/noisy and
        # selects the wrong checkpoint -- worst on small datasets. None => the
        # model was not given history (val masks special tokens only).
        self._val_hist_ids = None

        # Set up loss function
        self.loss_fn = None
        if hasattr(config, "loss_type") and config.loss_type:
            from ..losses.factory import get_loss_function

            # Determine model type based on class name
            implicit_models = ["NCF", "TwoTower", "BPRMF", "SimpleX"]
            model_type = (
                "implicit"
                if any(name in self.__class__.__name__ for name in implicit_models)
                else "sequential"
            )
            self.loss_fn = get_loss_function(config.loss_type, model_type=model_type)

    def configure_optimizers(self):
        """Configure optimizer for Lightning.

        Single AdamW for dense models. When any embedding table is sparse
        (config.sparse_embeddings on a sampled-loss implicit model), builds
        SparseAdam (tables) + AdamW (dense) wrapped in HybridOptim so Lightning
        stays in automatic optimization.
        """
        from ..utils.sparse_optim import build_optimizer

        weight_decay = self.config.get("weight_decay", 1e-2)
        optimizer = build_optimizer(self, lr=self.lr, weight_decay=weight_decay)

        # Add learning rate scheduler if specified
        scheduler_config = self.config.get("scheduler")
        if scheduler_config:
            scheduler_type = scheduler_config.get("type", "cosine")
            warmup_steps = scheduler_config.get("warmup_steps", 0)

            if scheduler_type == "cosine":
                if warmup_steps > 0:
                    from torch.optim.lr_scheduler import LambdaLR

                    def lr_lambda(step):
                        if step < warmup_steps:
                            return step / warmup_steps
                        else:
                            progress = (step - warmup_steps) / (
                                self.config.get("max_epochs", 100)
                                * self.config.get("steps_per_epoch", 100)
                                - warmup_steps
                            )
                            return 0.5 * (1 + torch.cos(torch.pi * progress))

                    scheduler = LambdaLR(optimizer, lr_lambda)
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.config.get("max_epochs", 100)
                    )
            elif scheduler_type == "reduce_on_plateau":
                # Honor the scheduler-config knobs instead of hardcoding
                # mode/patience/monitor. The harness builds this dict (see
                # scripts/benchmark/core.py::_scheduler_config) with the SAME
                # monitor/mode as early stopping (e.g. val_ndcg@10 / max), so LR
                # decay and stopping track one signal. The previous hardcoded
                # mode="min"/monitor="val_loss" silently diverged from the
                # sequential model path and from what the harness configured.
                plateau_mode = scheduler_config.get("mode", "min")
                plateau_monitor = scheduler_config.get("monitor", "val_loss")
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=plateau_mode,
                    patience=scheduler_config.get("patience", 5),
                    factor=scheduler_config.get("factor", 0.5),
                )
            else:
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": (
                        plateau_monitor
                        if scheduler_type == "reduce_on_plateau"
                        else None
                    ),
                    "interval": "step" if warmup_steps > 0 else "epoch",
                },
            }

        return optimizer

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """Clip gradients, skipping sparse embedding tables.

        Lightning's automatic clipping calls ``linalg_vector_norm`` which has no
        sparse kernel, so when a table is sparse we clip dense params only.
        Dense models defer to Lightning's default behavior.
        """
        if not getattr(self, "_sparse_embeddings", False):
            return super().configure_gradient_clipping(
                optimizer, gradient_clip_val, gradient_clip_algorithm
            )
        if gradient_clip_val:
            from ..utils.sparse_optim import clip_dense_grads_only

            clip_dense_grads_only(
                self, gradient_clip_val, gradient_clip_algorithm or "norm"
            )

    def set_eval_history(self, hist_ids):
        """Register a per-user TRAIN-history table for validation masking.

        hist_ids: LongTensor [num_users, Lmax] of 3-indexed item ids (pad 0).
        Used only during validation metric computation (see validation_step) so
        the monitored val NDCG masks already-seen items exactly like the test
        full-sort eval. Must contain TRAIN items only -- never the val target --
        or the target would be masked out of its own ranking.
        """
        self._val_hist_ids = hist_ids

    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning."""
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute metrics during validation
        if self.compute_val_metrics:
            user_ids = batch["user_id"]
            item_ids = batch["item_id"]

            # Get predictions for all items
            if hasattr(self, "get_user_embedding") and hasattr(
                self, "get_item_embedding"
            ):
                user_embs = self.get_user_embedding(user_ids)
                all_item_embs = self.get_item_embedding(
                    torch.arange(self.num_items, device=user_ids.device)
                )
                predictions = torch.matmul(user_embs, all_item_embs.t())
            else:
                batch_size = user_ids.size(0)
                all_items = torch.arange(self.num_items, device=user_ids.device)
                user_expanded = user_ids.unsqueeze(1).expand(-1, self.num_items)
                item_expanded = all_items.unsqueeze(0).expand(batch_size, -1)
                predictions = self.predict(
                    user_expanded.flatten(), item_expanded.flatten()
                )
                predictions = predictions.view(batch_size, self.num_items)

            # Mask special tokens (PAD/UNK/MASK = cols 0/1/2) and each user's
            # already-seen TRAIN items, mirroring the TEST full-sort protocol
            # (core.py) and RecBole's masked valid metric. Without this the val
            # NDCG that drives early stopping + best-checkpoint selection is a
            # deflated, noisy signal that disagrees with the reported test
            # metric -- selecting the wrong epoch, worst on small datasets.
            if predictions.size(1) >= 3:
                predictions[:, :3] = float("-inf")
            if self._val_hist_ids is not None:
                hist = self._val_hist_ids.to(predictions.device)[user_ids]
                hist = hist.clamp_(max=predictions.size(1) - 1)
                predictions.scatter_(1, hist, float("-inf"))
                predictions[:, 0] = float("-inf")  # undo PAD-col unmask (pad=0)

            # Compute metrics
            if self._metric_calculator is None:
                from rec_arena.metrics import MetricCalculator

                self._metric_calculator = MetricCalculator(k_values=self.val_k_values)

            predictions_cpu = predictions.detach().cpu()
            targets_cpu = item_ids.detach().cpu()
            metrics = self._metric_calculator.calculate_all(
                predictions_cpu, targets_cpu
            )

            for metric_name, value in metrics.items():
                self.log(
                    f"val_{metric_name}",
                    float(value),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        return loss

    def test_step(self, batch, batch_idx):
        """Test step for evaluation - computes and logs metrics.

        Note: item_ids are already 0-indexed from the implicit dataset,
        matching the 0-indexed prediction tensor and embeddings.

        Returns predictions and targets for consistency with sequential models.
        """
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]

        # Compute predictions for all items
        if hasattr(self, "get_user_embedding") and hasattr(self, "get_item_embedding"):
            user_embs = self.get_user_embedding(user_ids)
            all_item_embs = self.get_item_embedding(
                torch.arange(self.num_items, device=user_ids.device)
            )
            predictions = torch.matmul(user_embs, all_item_embs.t())
        else:
            # Fallback: compute predictions for all items per user
            batch_size = user_ids.size(0)
            all_items = torch.arange(self.num_items, device=user_ids.device)
            user_expanded = user_ids.unsqueeze(1).expand(-1, self.num_items)
            item_expanded = all_items.unsqueeze(0).expand(batch_size, -1)
            predictions = self.predict(user_expanded.flatten(), item_expanded.flatten())
            predictions = predictions.view(batch_size, self.num_items)

        # Compute accuracy@10
        _, top10 = torch.topk(predictions, k=10, dim=-1)
        acc10 = (top10 == item_ids.unsqueeze(1)).any(dim=1).float().mean()
        self.log(
            "test_acc@10", float(acc10), on_step=False, on_epoch=True, prog_bar=True
        )

        # Compute additional metrics
        if self._metric_calculator is None:
            from rec_arena.metrics import MetricCalculator

            self._metric_calculator = MetricCalculator(k_values=self.val_k_values)

        predictions_cpu = predictions.detach().cpu()
        targets_cpu = item_ids.detach().cpu()
        metrics = self._metric_calculator.calculate_all(predictions_cpu, targets_cpu)

        for metric_name, value in metrics.items():
            self.log(
                f"test_{metric_name}",
                float(value),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # Return predictions and targets for external metric calculation
        return {
            "predictions": predictions_cpu,
            "targets": targets_cpu,
            "test_acc@10": acc10,
        }

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch. To be implemented by subclasses."""
        raise NotImplementedError

    def fit(self, train_data, val_data=None) -> None:
        """Train using Lightning Trainer."""
        from lightning import Trainer
        from lightning.pytorch.callbacks import EarlyStopping

        logger.info(f"Starting training for {self.__class__.__name__}")

        callbacks = []

        # Early stopping
        if self.config.get("early_stopping", True):
            early_stop = EarlyStopping(
                monitor="val_loss", patience=self.config.get("patience", 10), mode="min"
            )
            callbacks.append(early_stop)

        trainer = Trainer(
            max_epochs=self.config.get("max_epochs", 100),
            accelerator="auto",
            devices="auto",
            strategy="auto",
            precision=self.config.get("precision", "16-mixed"),  # Mixed precision
            gradient_clip_val=self.gradient_clip_val,  # Gradient clipping
            callbacks=callbacks,
        )
        trainer.fit(self, train_data, val_data)
        logger.info("Training completed")

        # Clear memory after training
        clear_memory()

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        import os
        from pathlib import Path

        try:
            # Secure path validation
            path_obj = Path(path).resolve()
            if not str(path_obj).startswith(os.getcwd()):
                raise ValueError("Path must be within current directory")
            if ".." in str(path_obj) or not path.endswith((".pt", ".pth")):
                raise ValueError("Invalid path or file extension")

            os.makedirs(path_obj.parent, exist_ok=True)
            torch.save(self.state_dict(), path_obj)
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        import os
        from pathlib import Path

        try:
            # Secure path validation
            path_obj = Path(path).resolve()
            if not str(path_obj).startswith(os.getcwd()):
                raise ValueError("Path must be within current directory")
            if not path_obj.exists() or not path.endswith((".pt", ".pth")):
                raise ValueError("Invalid or non-existent path")

            # Safe deserialization
            self.load_state_dict(
                torch.load(path_obj, map_location="cpu", weights_only=True)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
