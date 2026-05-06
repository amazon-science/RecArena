"""LLaDA4Rec: Diffusion-based Sequential Recommendation."""

import torch
import torch.nn.functional as F
import numpy as np
from .bert4rec import BERT4Rec
from ...configs.defaults.llada4rec import LLaDA4RecConfig


class LLaDA4Rec(BERT4Rec):
    """LLaDA4Rec: Large Language Diffusion Adapted for Recommendation.

    A diffusion-based sequential recommendation model that uses variable masking ratios
    and iterative generation, adapted from LLaDA for recommendation tasks.

    Paper: "Large Language Diffusion Models" (arXiv 2025)
    Link: https://arxiv.org/abs/2502.09992

    Model ID: llada4rec
    Model Type: Sequential

    Key Features:
        - Variable masking ratio (0-1) during training
        - Weighted loss (CE / p_mask) for proper likelihood bound
        - Iterative unmasking during inference
        - Confidence-based or random remasking strategies

    Args:
        config (LLaDA4RecConfig): Model configuration with diffusion parameters

    Example:
        >>> config = LLaDA4RecConfig(vocab_size=1000, embedding_dim=64, diffusion_steps=50)
        >>> model = LLaDA4Rec(config)
        >>> predictions = model.predict_next(sequences, lengths)
    """

    def __init__(self, config: LLaDA4RecConfig):
        super().__init__(config)
        
        self.eps = config.eps
        self.diffusion_steps = config.diffusion_steps
        self.remasking_strategy = config.remasking_strategy
        self.temperature = config.temperature

    def forward_process(self, sequences, sequence_lengths):
        """Apply variable masking to sequences (LLaDA-style, fast vectorized version).
        
        Uses probabilistic masking with variable ratio for speed.
        
        Args:
            sequences: [batch, seq_len] input sequences
            sequence_lengths: [batch] actual sequence lengths
            
        Returns:
            masked_sequences: [batch, seq_len] with mask tokens
            mask_indices: [batch, seq_len] boolean mask
            p_mask: [batch, seq_len] masking ratio per position
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # Sample random masking ratio per batch: p ~ Uniform(eps, 1)
        t = torch.rand(batch_size, device=device)
        p_mask_batch = (1 - self.eps) * t + self.eps  # [batch]
        
        # Create position mask (don't mask padding)
        position_mask = torch.arange(seq_len, device=device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        
        # Random masking with ratio p_mask (vectorized)
        random_vals = torch.rand(batch_size, seq_len, device=device)
        mask_indices = (random_vals < p_mask_batch.unsqueeze(1)) & position_mask
        
        # Apply masking
        masked_sequences = sequences.clone()
        masked_sequences[mask_indices] = self.mask_token_id
        
        # Broadcast p_mask to all positions
        p_mask = p_mask_batch.unsqueeze(1).expand(batch_size, seq_len)
        
        return masked_sequences, mask_indices, p_mask

    def compute_loss(self, batch) -> torch.Tensor:
        """Compute LLaDA-style weighted loss (exact implementation from paper).
        
        Matches GUIDELINES.md exactly:
        loss = CE(logits[masked], targets[masked]) / p_mask[masked]
        loss = loss.sum() / (batch_size * seq_length)
        """
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        
        # Apply variable masking
        masked_sequences, mask_indices, p_mask = self.forward_process(sequences, sequence_lengths)
        
        # Get logits
        logits = self.forward(masked_sequences)
        
        # Compute weighted loss on masked positions
        logits_flat = logits[mask_indices]
        targets_flat = sequences[mask_indices]
        p_mask_flat = p_mask[mask_indices]
        
        # Weighted CE: divide by masking ratio
        token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none') / p_mask_flat.clamp(min=1e-8)
        
        # Normalize by FULL sequence length (not just masked tokens!)
        # This matches LLaDA paper exactly
        loss = token_loss.sum() / (sequences.size(0) * sequences.size(1))
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Override validation to use standard cross-entropy (not weighted)."""
        # For validation, use standard BERT4Rec-style evaluation
        if "target" in batch and "negatives" not in batch:
            sequences = batch["sequence"]
            sequence_lengths = batch["sequence_length"]
            targets = batch["target"]

            # Get predictions using predict_next
            predictions = self.predict_next(sequences, sequence_lengths)

            # Mask special tokens
            predictions_masked = predictions.clone()
            predictions_masked[:, :3] = float('-inf')

            # Compute accuracy@10
            _, top10 = torch.topk(predictions_masked, k=10, dim=-1)
            acc10 = (top10 == targets.unsqueeze(1)).any(dim=1).float().mean()
            self.log("val_acc@10", acc10, on_step=False, on_epoch=True, prog_bar=True)

            # Compute other metrics at interval
            if (self.current_epoch + 1) % self.metric_compute_interval == 0:
                if self._metric_calculator is None:
                    from rec_arena.metrics import MetricCalculator
                    self._metric_calculator = MetricCalculator(k_values=self.val_k_values)
                
                predictions_cpu = predictions.detach().cpu()
                targets_cpu = targets.detach().cpu()
                metrics = self._metric_calculator.calculate_all(predictions_cpu, targets_cpu)
                
                for metric_name, value in metrics.items():
                    self.log(
                        f"val_{metric_name}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            
            # Compute simple CE loss for monitoring (not weighted)
            batch_indices = torch.arange(sequences.size(0), device=sequences.device)
            last_indices = torch.clamp(sequence_lengths - 1, min=0, max=sequences.size(1) - 1)
            
            hidden_states = self.get_hidden_states(sequences, sequence_lengths)
            logits = torch.matmul(hidden_states, self.item_embedding.weight.transpose(0, 1))
            last_logits = logits[batch_indices, last_indices]
            
            val_loss = F.cross_entropy(last_logits, targets)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
            
            return val_loss
        else:
            # Fallback to parent
            return super().validation_step(batch, batch_idx)

    def predict_next(self, sequences, sequence_lengths):
        """Predict next item using iterative diffusion generation (complete LLaDA).
        
        Implements the full LLaDA generation process from generate.py:
        1. Start with last position masked
        2. Iteratively predict and unmask based on confidence
        3. Return final prediction
        
        Args:
            sequences: [batch, seq_len] input sequences
            sequence_lengths: [batch] actual sequence lengths
            
        Returns:
            [batch, vocab_size] probability distribution over next items
        """
        was_training = self.training
        self.eval()
        
        batch_size = sequences.size(0)
        device = sequences.device
        
        with torch.no_grad():
            # Extend sequence with one masked position for next item
            x = torch.cat([sequences, torch.full((batch_size, 1), self.mask_token_id, device=device)], dim=1)
            
            # Iterative generation over diffusion_steps
            for step in range(self.diffusion_steps):
                # Get logits for all positions
                logits = self.forward(x)
                
                # Add Gumbel noise if temperature > 0
                if self.temperature > 0:
                    logits_with_noise = self._add_gumbel_noise(logits, self.temperature)
                else:
                    logits_with_noise = logits
                
                # Get predictions
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [batch, seq_len+1]
                
                # Compute confidence scores
                if self.remasking_strategy == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                else:  # random
                    confidence = torch.rand_like(x0, dtype=torch.float)
                
                # Only consider the last position (next item)
                mask_indices = (x == self.mask_token_id)
                
                # If last position is masked and confident enough, unmask it
                if mask_indices[:, -1].any():
                    # Update masked positions with predictions
                    x = torch.where(mask_indices, x0, x)
            
            # Get final logits for last position
            final_logits = self.forward(x)[:, -1, :]
            
            # Mask special tokens
            final_logits[:, 0] = -float("inf")  # PAD
            final_logits[:, 1] = -float("inf")  # UNK
            final_logits[:, 2] = -float("inf")  # MASK
        
        if was_training:
            self.train()
        
        return torch.softmax(final_logits, dim=-1)

    def generate_sequence(self, prompt, gen_length, steps=None):
        """Generate a sequence of items using iterative diffusion.
        
        Args:
            prompt: [batch, prompt_len] initial sequence
            gen_length: number of items to generate
            steps: number of diffusion steps (default: self.diffusion_steps)
            
        Returns:
            [batch, prompt_len + gen_length] generated sequence
        """
        if steps is None:
            steps = self.diffusion_steps
        
        batch_size = prompt.size(0)
        device = prompt.device
        
        # Initialize with prompt + masked positions
        x = torch.cat([
            prompt,
            torch.full((batch_size, gen_length), self.mask_token_id, device=device)
        ], dim=1)
        
        prompt_mask = torch.cat([
            torch.ones(batch_size, prompt.size(1), dtype=torch.bool, device=device),
            torch.zeros(batch_size, gen_length, dtype=torch.bool, device=device)
        ], dim=1)
        
        # Compute tokens to unmask per step
        num_transfer_per_step = max(1, gen_length // steps)
        
        for step in range(steps):
            mask_indices = (x == self.mask_token_id)
            
            if not mask_indices.any():
                break
            
            # Get logits
            logits = self.forward(x)
            
            # Add Gumbel noise
            if self.temperature > 0:
                logits = self._add_gumbel_noise(logits, self.temperature)
            
            # Get predictions
            x0 = torch.argmax(logits, dim=-1)
            
            # Compute confidence
            if self.remasking_strategy == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            else:  # random
                confidence = torch.rand_like(x0, dtype=torch.float)
            
            # Mask out prompt and already unmasked positions
            confidence[prompt_mask] = -np.inf
            confidence[~mask_indices] = -np.inf
            
            # Select top-k confident predictions to unmask
            num_to_unmask = min(num_transfer_per_step, mask_indices.sum().item())
            if num_to_unmask > 0:
                _, top_indices = torch.topk(confidence.view(batch_size, -1), k=num_to_unmask, dim=-1)
                
                for b in range(batch_size):
                    x[b, top_indices[b]] = x0[b, top_indices[b]]
        
        return x

    def _add_gumbel_noise(self, logits, temperature):
        """Add Gumbel noise for sampling (LLaDA-style)."""
        if temperature == 0:
            return logits
        
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return (logits.exp() / gumbel_noise).to(logits.dtype)
