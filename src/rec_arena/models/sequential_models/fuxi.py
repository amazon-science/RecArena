import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..sequential import DeepSequentialModel
from ...configs.defaults.fuxi import FuXiConfig


class SeparatedRelativeBucketedTimeAndPositionBasedBias(nn.Module):
    """Separated relative position and time bias."""

    def __init__(self, max_seq_len: int, num_buckets: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self._ts_w = nn.Parameter(torch.empty(num_buckets + 1).normal_(mean=0, std=0.02))
        self._pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02))
        self._num_buckets = num_buckets

    def forward(self, all_timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = all_timestamps.size(0)
        N = self.max_seq_len
        
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        pos_bias = t[:, :, r:-r]
        
        ext_timestamps = torch.cat([all_timestamps, all_timestamps[:, N - 1 : N]], dim=1)
        bucketed_timestamps = torch.clamp(
            (torch.log(torch.abs(ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)).clamp(min=1)) / 0.301).long(),
            min=0,
            max=self._num_buckets,
        ).detach()
        ts_bias = torch.index_select(self._ts_w, dim=0, index=bucketed_timestamps.view(-1)).view(B, N, N)
        
        return pos_bias, ts_bias


class MultistageFeedforwardNeuralNetwork(nn.Module):
    """Multi-stage FFN with gated linear units and RMSNorm."""

    def __init__(self, ams_output_size: int, input_size: int, hidden_size: int, 
                 output_size: int, dropout_ratio: float, single_stage: bool = False, epsilon: float = 1e-6):
        super().__init__()
        self.lin0 = nn.Linear(ams_output_size, input_size)
        self.is_single_stage = single_stage
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        
        if not single_stage:
            self.lin1 = nn.Linear(input_size, hidden_size)
            self.lin2 = nn.Linear(hidden_size, output_size)
            self.lin3 = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        x = self.lin0(F.dropout(x, p=self.dropout_ratio, training=self.training)) + x0
        
        if not self.is_single_stage:
            normed_x = F.rms_norm(x, normalized_shape=[self.input_size], eps=self.eps)
            normed_x = F.dropout(normed_x, p=self.dropout_ratio, training=self.training)
            x1 = F.silu(self.lin1(normed_x)) * self.lin3(normed_x)
            x = self.lin2(x1) + x
        
        return x
    
    def init(self):
        torch.nn.init.xavier_uniform_(self.lin0.weight)
        if not self.is_single_stage:
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            torch.nn.init.xavier_uniform_(self.lin3.weight)


class FuXiBlock(nn.Module):
    """FuXi block with adaptive multi-channel attention and multi-stage FFN."""

    def __init__(self, embedding_dim: int, linear_hidden_dim: int, attention_dim: int,
                 dropout_ratio: float, num_heads: int, max_seq_len: int,
                 linear_activation: str = "silu", ffn_multiply: float = 1.0, 
                 ffn_single_stage: bool = False, epsilon: float = 1e-6):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_hidden_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.eps = epsilon
        self.linear_activation = linear_activation
        
        self._uvqk = nn.Parameter(torch.empty(
            embedding_dim,
            linear_hidden_dim * num_heads * 3 + linear_hidden_dim * num_heads + attention_dim * num_heads * 2
        ).normal_(mean=0, std=0.02))
        
        self.rel_bias = SeparatedRelativeBucketedTimeAndPositionBasedBias(max_seq_len, 128)
        
        self.mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size=linear_hidden_dim * num_heads * 3,
            input_size=embedding_dim,
            hidden_size=int(embedding_dim * ffn_multiply),
            output_size=embedding_dim,
            dropout_ratio=dropout_ratio,
            single_stage=ffn_single_stage,
            epsilon=epsilon
        )
        self.mffn.init()
        
        self.norm_input_shape = embedding_dim
        self.norm_attn_shape = linear_hidden_dim * num_heads * 3

    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        
        normed_x = F.rms_norm(x, normalized_shape=[self.norm_input_shape], eps=self.eps)
        proj = torch.mm(normed_x.view(-1, D), self._uvqk).view(B, N, -1)
        if self.linear_activation == "silu":
            proj = F.silu(proj)
        
        u, v, q, k = torch.split(proj, [
            self.linear_dim * self.num_heads * 3,
            self.linear_dim * self.num_heads,
            self.attention_dim * self.num_heads,
            self.attention_dim * self.num_heads,
        ], dim=-1)
        
        q = q.view(B, N, self.num_heads, self.attention_dim)
        k = k.view(B, N, self.num_heads, self.attention_dim)
        v = v.view(B, N, self.num_heads, self.linear_dim)
        
        # CRITICAL: einsum must produce [B, H, N, M] not [B, N, M, H]
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k)
        qk_attn = F.silu(qk_attn) / N
        
        pos_bias, ts_bias = self.rel_bias(timestamps)
        
        if causal_mask is not None:
            invalid_mask = (~causal_mask).float()
            pos_bias = pos_bias * invalid_mask.unsqueeze(0)
            ts_bias = ts_bias * invalid_mask.unsqueeze(0)
            qk_attn = qk_attn * invalid_mask.unsqueeze(0).unsqueeze(0)
        
        # All attention outputs: [B, N, H, D]
        output_pos = torch.einsum("bnm,bmhd->bnhd", pos_bias, v)
        output_ts = torch.einsum("bnm,bmhd->bnhd", ts_bias, v)
        output_latent = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v)
        
        combined = torch.cat([output_pos, output_ts, output_latent], dim=-1).view(B, N, -1)
        
        ams_output = u * F.rms_norm(combined, normalized_shape=[self.norm_attn_shape], eps=self.eps)
        
        return self.mffn(ams_output, x)


class FuXi(DeepSequentialModel):
    """FuXi-α: Scaling Recommendation Model with Feature Interaction Enhanced Transformer."""

    def __init__(self, config: FuXiConfig):
        super().__init__(config)
        self.save_hyperparameters()
        self.epsilon = config.get("epsilon", 1e-6)

        self.pos_embedding = nn.Embedding(config.max_seq_length, self.embedding_dim)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        self.fuxi_blocks = nn.ModuleList([
            FuXiBlock(
                self.embedding_dim,
                config.linear_dim,
                config.attention_dim,
                config.dropout_rate,
                config.num_heads,
                config.max_seq_length,
                getattr(config, "linear_activation", "silu"),
                config.ffn_multiply,
                config.ffn_single_stage,
                self.epsilon
            )
            for _ in range(config.num_layers)
        ])

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_length, config.max_seq_length, dtype=torch.bool), diagonal=1)
        )
        self._init_weights()

    def get_hidden_states(self, sequences: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(sequences)
        positions = torch.arange(sequences.size(1), device=sequences.device)
        x = x + self.pos_embedding(positions)
        
        timestamps = positions.unsqueeze(0).expand(sequences.size(0), -1).float()
        
        for block in self.fuxi_blocks:
            x = block(x, timestamps=timestamps, causal_mask=self.causal_mask)
        
        return x

    def predict_next(self, sequences: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        logits = torch.matmul(hidden_states, self.item_embedding.weight.transpose(0, 1))
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(sequence_lengths - 1, min=0, max=sequences.size(1) - 1)
        return torch.softmax(logits[batch_indices, last_indices], dim=-1)

    def get_sequence_embedding(self, sequences: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
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
            # uvqk already initialized in __init__
            block.mffn.init()  # Uses xavier_uniform for FFN weights
