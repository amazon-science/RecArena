import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..sequential import DeepSequentialModel
from ...configs.defaults.fuxi_gamma import FuXiGammaConfig
from ...modules.layer_utils.swiglu import SwiGLU


class TemporalPositionalBias(nn.Module):
    def __init__(self, max_seq_len, range_alpha=0.1, left_beta=0.5, right_beta=2.0, 
                 gamma_learnable=True, left_gamma=0.5, right_gamma=0.99):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.epsilon = 1e-6
        
        self._alpha = nn.Parameter(torch.empty(1).uniform_(-range_alpha, range_alpha))
        self._beta = nn.Parameter(torch.empty(1).uniform_(left_beta, right_beta))
        if gamma_learnable:
            self._gamma = nn.Parameter(torch.empty(1).uniform_(left_gamma, right_gamma))
        else:
            self.register_parameter('_gamma', nn.Parameter(torch.tensor(0.8), requires_grad=False))
        
        self._pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02))

    def forward(self, relative_temporal_intervals):
        ts_bias = self._alpha * torch.pow(self._gamma, torch.pow(relative_temporal_intervals, self._beta))
        
        N = self.max_seq_len
        t = F.pad(self._pos_w[:2*N-1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3*N-2)
        r = (2*N-1) // 2
        pos_bias = t[:, :, r:-r]
        
        return ts_bias, pos_bias


class FuXiGammaBlock(nn.Module):
    def __init__(self, embedding_dim, linear_dim, attention_dim, dropout_ratio, 
                 num_heads, max_seq_len, linear_activation="silu", ffn_multiply=1.0, 
                 range_alpha=0.1, left_beta=0.5, right_beta=2.0, gamma_learnable=True,
                 left_gamma=0.5, right_gamma=0.99, epsilon=1e-6):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.num_heads = num_heads
        self.linear_activation = linear_activation
        self.epsilon = epsilon
        self.norm_input_shape = embedding_dim
        self.norm_attn_shape = embedding_dim * 2  # Combined output is embedding_dim * 2
        self.norm_ffn_shape = embedding_dim
        
        self._uv = nn.Parameter(torch.empty(embedding_dim, embedding_dim * 3).normal_(mean=0, std=0.02))
        
        self.rel_bias = TemporalPositionalBias(max_seq_len, range_alpha, left_beta, right_beta,
                                               gamma_learnable, left_gamma, right_gamma)
        
        self._o = nn.Linear(embedding_dim * 2, embedding_dim)
        nn.init.xavier_uniform_(self._o.weight)
        
        self.swiglu = SwiGLU(embedding_dim, int(embedding_dim * ffn_multiply), dropout_ratio, bias=False)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, timestamps=None, causal_mask=None):
        B, N, D = x.shape
        
        normed_x = F.rms_norm(x, normalized_shape=[self.norm_input_shape], eps=self.epsilon)
        proj = torch.mm(normed_x.view(-1, D), self._uv).view(B, N, -1)
        if self.linear_activation == "silu":
            proj = F.silu(proj)
        
        u, v = torch.split(proj, [self.linear_dim * self.num_heads * 2, self.linear_dim * self.num_heads], dim=-1)
        
        ext_ts = torch.cat([timestamps, timestamps[:, -1:]], dim=1)
        rel_ts = (ext_ts[:, 1:].unsqueeze(2) - ext_ts[:, :-1].unsqueeze(1)).abs()
        
        ts_bias, pos_bias = self.rel_bias(rel_ts)
        
        if causal_mask is not None:
            mask = (~causal_mask).float()
            ts_bias = ts_bias * mask.unsqueeze(0)
            pos_bias = pos_bias * mask.unsqueeze(0)
        
        output_ts = torch.einsum("bnm,bmd->bnd", ts_bias, v)
        output_pos = torch.einsum("bnm,bmd->bnd", pos_bias, v)
        
        combined = torch.cat([output_ts, output_pos], dim=-1)
        
        o_input = u * F.rms_norm(combined, normalized_shape=[self.norm_attn_shape], eps=self.epsilon)
        o_output = self._o(self.dropout(o_input)) + x
        
        ffn_input = F.rms_norm(o_output, normalized_shape=[self.norm_ffn_shape], eps=self.epsilon)
        output = self.dropout(self.swiglu(ffn_input)) + o_output
        
        return output


class FuXiGamma(DeepSequentialModel):
    """FuXi-γ: Exponential-Power Temporal Encoder with SwiGLU FFN."""

    def __init__(self, config: FuXiGammaConfig):
        super().__init__(config)
        self.save_hyperparameters()
        self.epsilon = config.get("epsilon", 1e-6)

        self.pos_embedding = nn.Embedding(config.max_seq_length, self.embedding_dim)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        self.fuxi_blocks = nn.ModuleList([
            FuXiGammaBlock(
                self.embedding_dim,
                config.linear_dim,
                config.attention_dim,
                config.dropout_rate,
                config.num_heads,
                config.max_seq_length,
                getattr(config, "linear_activation", "silu"),
                config.ffn_multiply,
                getattr(config, "range_alpha", 0.1),
                getattr(config, "left_beta", 0.5),
                getattr(config, "right_beta", 2.0),
                getattr(config, "gamma_learnable", True),
                getattr(config, "left_gamma", 0.5),
                getattr(config, "right_gamma", 0.99),
                self.epsilon
            )
            for _ in range(config.num_layers)
        ])

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_length, config.max_seq_length, dtype=torch.bool), diagonal=1)
        )
        self._init_weights()

    def get_hidden_states(self, sequences, sequence_lengths):
        x = self.item_embedding(sequences)
        positions = torch.arange(sequences.size(1), device=sequences.device)
        x = x + self.pos_embedding(positions)
        
        timestamps = positions.unsqueeze(0).expand(sequences.size(0), -1).float()
        
        for block in self.fuxi_blocks:
            x = block(x, timestamps=timestamps, causal_mask=self.causal_mask)
        
        return x

    def predict_next(self, sequences, sequence_lengths):
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        logits = torch.matmul(hidden_states, self.item_embedding.weight.transpose(0, 1))
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0, max=sequences.size(1) - 1)
        return torch.softmax(logits[batch_indices, last_indices], dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0, max=hidden_states.size(1) - 1)
        return hidden_states[batch_indices, last_indices]

    def get_targets_and_mask(self, batch):
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        targets = sequences
        seq_len = sequences.size(1)
        
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        mask = (positions >= 1) & (positions < sequence_lengths.unsqueeze(1))
        
        return targets, mask

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        for block in self.fuxi_blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
