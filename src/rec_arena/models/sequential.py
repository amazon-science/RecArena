import torch
import torch.nn as nn
import lightning as pl
import numpy as np
from abc import abstractmethod
from typing import Dict, Any, Tuple, Optional
from .base import BaseModel
from ..configs.validation import validate_config


class SequentialModel(BaseModel):
    """Abstract base class for sequential recommendation models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_seq_length = config.get("max_seq_length", 50)
        self.vocab_size = config.get("vocab_size")

        if self.vocab_size is None or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")

    @abstractmethod
    def predict_next(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Predict next item probabilities for sequences.

        Args:
            sequences: [batch_size, max_seq_length] padded sequences
            sequence_lengths: [batch_size] actual sequence lengths

        Returns:
            [batch_size, num_items] next item probabilities
        """
        pass

    @abstractmethod
    def recommend_next(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k next item recommendations.

        Returns:
            Tuple of (recommended_items, scores) each [batch_size, k]
        """
        pass

    @abstractmethod
    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific items.

        Args:
            item_ids: [batch_size] or [batch_size, seq_len] item IDs

        Returns:
            [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim] embeddings
        """
        pass

    @abstractmethod
    def get_sequence_embedding(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Get sequence-level embeddings.

        Args:
            sequences: [batch_size, max_seq_length] padded sequences
            sequence_lengths: [batch_size] actual sequence lengths

        Returns:
            [batch_size, embedding_dim] sequence embeddings
        """
        pass

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Not applicable for sequential models."""
        raise NotImplementedError("Use predict_next() for sequential models")

    def recommend(
        self, user_ids: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Not applicable for sequential models."""
        raise NotImplementedError("Use recommend_next() for sequential models")


class DeepSequentialModel(SequentialModel, pl.LightningModule):
    """Base class for deep learning sequential models using PyTorch Lightning."""

    def __init__(self, config: Dict[str, Any]):
        # Validate configuration for sequential models
        validate_config(config, "sequential")

        SequentialModel.__init__(self, config)
        pl.LightningModule.__init__(self)

        self.embedding_dim = config.get("embedding_dim", 64)
        self.lr = config.get("lr", 1e-3)
        self.gradient_clip_val = config.get("gradient_clip_val", 1.0)
        # GPT-style special tokens
        self.pad_token = 0  # PAD
        self.unk_token = 1  # UNK
        self.mask_token = 2  # MASK (for BERT4Rec)
        # Items start at index 3

        # Optional metrics during validation
        self.compute_val_metrics = config.get("compute_val_metrics", False)
        self.val_k_values = config.get("val_k_values", [10])
        self.metric_compute_interval = config.get("metric_compute_interval", 10)
        self._metric_calculator = None
        self._last_val_ndcg = 0.0  # Cache last computed NDCG for early stopping
        self._compute_metrics_this_epoch = False  # Flag to track if we computed metrics

        # Loss function (can be set externally or from config)
        self.loss_fn = config.get("loss_fn", None)

        # If no external loss_fn, try to create from loss_type
        if self.loss_fn is None and hasattr(config, "loss_type"):
            from ..losses.factory import get_loss_function

            loss_kwargs = getattr(config, "loss_kwargs", {})
            self.loss_fn = get_loss_function(
                config.loss_type, model_type="sequential", **loss_kwargs
            )

        # Item embedding layer (vocab_size includes special tokens)
        embedding_config = config.get("embedding_config", {"type": "standard"})
        if embedding_config["type"] == "standard":
            self.item_embedding = nn.Embedding(
                self.vocab_size,
                self.embedding_dim,
                padding_idx=0,
            )
        else:
            from ..modules.layer_utils.embedding_factory import create_embedding
            self.item_embedding = create_embedding(
                embedding_type=embedding_config["type"],
                num_embeddings=self.vocab_size,
                embedding_dim=self.embedding_dim,
                padding_idx=0,
                **embedding_config.get("kwargs", {})
            )

    def _to_model_indices(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert 1-indexed item IDs to 0-indexed model indices.

        Dataset items are stored as [1, 2, 3, ...] but model predictions
        and embeddings use 0-indexed arrays [0, 1, 2, ...]. This method
        handles the conversion consistently across training and evaluation.

        Args:
            targets: Tensor of 1-indexed item IDs

        Returns:
            Tensor of 0-indexed model indices
        """
        return targets - 1

    def configure_optimizers(self):
        """Configure optimizer for Lightning."""
        weight_decay = self.config.get("weight_decay", 1e-6)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=weight_decay
        )

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
                monitor = scheduler_config.get("monitor", "val_loss")
                mode = scheduler_config.get("mode", "min")
                patience = scheduler_config.get("patience", 5)
                factor = scheduler_config.get("factor", 0.5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=mode, patience=patience, factor=factor
                )
            else:
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": (
                        scheduler_config.get("monitor", "val_loss") if scheduler_type == "reduce_on_plateau" else None
                    ),
                    "interval": "step" if warmup_steps > 0 else "epoch",
                },
            }

        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning."""
        # For LOO validation with explicit targets
        if "target" in batch and "negatives" not in batch:
            sequences = batch["sequence"]
            sequence_lengths = batch["sequence_length"]
            targets = batch["target"]

            # Compute loss only on target (last position)
            # Get logits for last position
            hidden_states = self.get_hidden_states(sequences, sequence_lengths)
            logits = torch.matmul(hidden_states, self.get_output_embeddings().transpose(0, 1))
            
            # Extract last position logits
            batch_indices = torch.arange(sequences.size(0), device=sequences.device)
            last_indices = torch.clamp(sequence_lengths - 1, min=0, max=sequences.size(1) - 1)
            last_logits = logits[batch_indices, last_indices]
            
            # Create single-position batch for loss function
            last_logits_expanded = last_logits.unsqueeze(1)  # [batch, 1, vocab]
            targets_expanded = targets.unsqueeze(1)  # [batch, 1]
            mask = torch.ones_like(targets_expanded, dtype=torch.bool)  # [batch, 1]
            
            # Extract negatives for last position if present
            neg_items = batch.get("neg_items")
            if neg_items is not None and neg_items.dim() == 3:
                # Per-position negatives: [batch, seq_len, num_neg] -> [batch, 1, num_neg]
                neg_items = neg_items[batch_indices, last_indices].unsqueeze(1)
            
            # Use model's loss function
            val_loss = self.loss_fn(
                logits=last_logits_expanded,
                targets=targets_expanded,
                mask=mask,
                neg_items=neg_items
            )
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

            # Only compute expensive full-vocab metrics every N epochs
            if (self.current_epoch + 1) % self.metric_compute_interval == 0:
                self._compute_metrics_this_epoch = True
                predictions = self.predict_next(sequences, sequence_lengths)
                predictions[:, :3] = float('-inf')  # Mask special tokens
                
                if self._metric_calculator is None:
                    from rec_arena.metrics import MetricCalculator
                    self._metric_calculator = MetricCalculator(k_values=self.val_k_values)
                
                metrics = self._metric_calculator.calculate_all(
                    predictions.detach().cpu(), targets.detach().cpu()
                )
                for metric_name, value in metrics.items():
                    self.log(f"val_{metric_name}", value, on_step=False, on_epoch=True, prog_bar=True)
            else:
                # Reuse last computed NDCG for early stopping
                self.log("val_ndcg@10", self._last_val_ndcg, on_step=False, on_epoch=True, prog_bar=True)
            
            return val_loss
        else:
            loss = self.compute_loss(batch)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def on_validation_epoch_end(self):
        """Cache the epoch-level NDCG after Lightning computes the average."""
        if self._compute_metrics_this_epoch:
            # Get the actual logged value (averaged across batches)
            if "val_ndcg@10" in self.trainer.callback_metrics:
                self._last_val_ndcg = self.trainer.callback_metrics["val_ndcg@10"].item()
            self._compute_metrics_this_epoch = False

    def test_step(self, batch, batch_idx):
        """Test step for evaluation - computes and logs metrics.
        
        Always returns predictions and targets for consistency with implicit models.
        """
        if "target" in batch:
            # Test data with explicit targets (like validation)
            sequences = batch["sequence"]
            sequence_lengths = batch["sequence_length"]
            targets = batch["target"]

            # Get predictions using predict_next method
            predictions = self.predict_next(sequences, sequence_lengths)

            # Mask special tokens (0, 1, 2) before computing metrics
            predictions_masked = predictions.clone()
            predictions_masked[:, :3] = float('-inf')

            # Compute accuracy@10
            _, top10 = torch.topk(predictions_masked, k=10, dim=-1)
            acc10 = (top10 == targets.unsqueeze(1)).any(dim=1).float().mean()
            self.log("test_acc@10", acc10, on_step=False, on_epoch=True, prog_bar=True)

            # Compute additional metrics
            if self._metric_calculator is None:
                from rec_arena.metrics import MetricCalculator

                self._metric_calculator = MetricCalculator(k_values=self.val_k_values)

            predictions_cpu = predictions_masked.detach().cpu()
            targets_cpu = targets.detach().cpu()
            metrics = self._metric_calculator.calculate_all(
                predictions_cpu, targets_cpu
            )

            for metric_name, value in metrics.items():
                self.log(
                    f"test_{metric_name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

            # Return predictions and targets for external metric calculation
            return {
                "predictions": predictions_cpu,
                "targets": targets_cpu,
                "test_acc@10": acc10
            }
        else:
            # Fallback for old-style test data
            sequences = batch["sequence"]
            sequence_lengths = batch["sequence_length"]

            batch_indices = torch.arange(sequences.size(0), device=sequences.device)
            last_indices = torch.clamp(sequence_lengths - 1, min=0)
            targets = sequences[batch_indices, last_indices]

            eval_sequences = sequences.clone()
            eval_lengths = torch.clamp(sequence_lengths - 1, min=1)

            for i, length in enumerate(sequence_lengths):
                if length > 1:
                    eval_sequences[i, length - 1] = 0

            predictions = self.predict_next(eval_sequences, eval_lengths)

            return {
                "predictions": predictions.detach().cpu(),
                "targets": targets.detach().cpu(),
            }

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute loss using loss function with proper signature.

        Loss functions expect: loss_fn(logits, targets, mask, neg_items=None)
        OR for sampled losses: loss_fn(hidden_states, item_embeddings, targets, mask, neg_items)

        This method handles:
        1. Computing hidden states
        2. Applying causal shift (predict position i+1 from position i)
        3. Extracting and preparing negative samples if present
        4. Creating appropriate loss masks
        5. Computing logits (full vocab for cross_entropy, sampled for others)

        Models can override this for special cases (BERT4Rec masking, etc.)
        """
        if self.loss_fn is None:
            raise RuntimeError(
                "No loss function set. Either pass loss_fn in config or use external loss functions."
            )

        # Get hidden states
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)

        # Get targets and mask from model
        targets, mask = self.get_targets_and_mask(batch)

        # Apply causal shift: predict next item
        # Logit at position i should predict item at position i+1
        shifted_hidden = hidden_states[:, :-1, :]  # Positions 0 to T-1
        shifted_targets = targets[:, 1:]  # Positions 1 to T
        shifted_mask = mask[:, 1:]  # Positions 1 to T

        # Extract and prepare negative samples if present
        neg_items = batch.get("neg_items")
        if neg_items is not None:
            # Shift negatives to match shifted targets
            if neg_items.dim() == 3 and neg_items.size(1) == targets.size(1):
                # Per-position negatives: [batch, seq_len, num_neg]
                neg_items = neg_items[:, 1:, :]  # Shift to [batch, seq_len-1, num_neg]
            elif neg_items.dim() == 2 and neg_items.size(1) == targets.size(1):
                # Per-position single neg: [batch, seq_len]
                neg_items = neg_items[:, 1:]  # Shift to [batch, seq_len-1]
            # else: assume global negatives per batch, no shifting needed

        # Call loss function with hidden states for fast sampled logits
        # Loss functions will use fast path if they support it, otherwise fall back to full logits
        return self.loss_fn(
            hidden_states=shifted_hidden,
            item_embeddings=self.get_output_embeddings(),
            targets=shifted_targets,
            mask=shifted_mask,
            neg_items=neg_items,
        )

    def set_loss_fn(self, loss_fn):
        """Set loss function for this model."""
        self.loss_fn = loss_fn

    def recommend_next(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default implementation using predict_next."""
        probs = self.predict_next(sequences, sequence_lengths)
        scores, items = torch.topk(probs, k, dim=-1)
        return items, scores

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specific items."""
        # For now, use item_ids directly (bypass tokenizer for simplicity)
        # Clamp to valid embedding range
        clamped_ids = torch.clamp(item_ids, 0, self.item_embedding.num_embeddings - 1)
        return self.item_embedding(clamped_ids)

    def get_output_embeddings(self) -> torch.Tensor:
        """Get output embedding weights for logit computation.
        
        Override this in subclasses for untied embeddings or LoRA.
        Returns: [vocab_size, embedding_dim] tensor
        """
        return self.item_embedding.weight

    def get_sequence_embedding(
        self, sequences: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Get sequence-level embeddings (last valid hidden state)."""
        # This is a default implementation - subclasses should override for model-specific logic
        embedded = self.get_item_embedding(sequences)  # [batch_size, seq_len, emb_dim]

        # Get last valid embedding for each sequence
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        # Clamp to avoid negative indices from zero-length sequences
        last_indices = torch.clamp(sequence_lengths - 1, min=0)

        return embedded[batch_indices, last_indices]  # [batch_size, emb_dim]

    def fit(self, train_data, val_data=None) -> None:
        """Train using Lightning Trainer."""
        from lightning import Trainer
        from lightning.pytorch.callbacks import EarlyStopping

        callbacks = []

        # Early stopping
        if self.config.get("early_stopping", True):
            early_stop = EarlyStopping(
                monitor="val_loss", patience=self.config.get("patience", 10), mode="min"
            )
            callbacks.append(early_stop)

        trainer = Trainer(
            max_epochs=self.config.get("max_epochs", 30),
            accelerator="auto",
            devices="auto",
            strategy="auto",
            precision=self.config.get("precision", "16-mixed"),  # Mixed precision
            gradient_clip_val=self.gradient_clip_val,  # Gradient clipping
            callbacks=callbacks,
        )
        trainer.fit(self, train_data, val_data)

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

    def _get_activation(self):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(self.config.activation, nn.ReLU())
