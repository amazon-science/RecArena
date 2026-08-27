import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..sequential import DeepSequentialModel
from ...configs.defaults.fuxi_gamma import FuXiGammaConfig
from ...modules.layer_utils.swiglu import SwiGLU
from ...modules.transformer_layers.transformer_block import RMSNorm


def truncated_normal(
    x: torch.Tensor, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """In-place truncated normal init (mirrors the FuXi reference util)."""
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
    return x


class TemporalPositionalBias(nn.Module):
    def __init__(
        self,
        max_seq_len,
        range_alpha=0.1,
        left_beta=0.5,
        right_beta=2.0,
        gamma_learnable=True,
        left_gamma=0.5,
        right_gamma=0.99,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.epsilon = 1e-6

        self._alpha = nn.Parameter(torch.empty(1).uniform_(-range_alpha, range_alpha))
        self._beta = nn.Parameter(torch.empty(1).uniform_(left_beta, right_beta))
        if gamma_learnable:
            self._gamma = nn.Parameter(torch.empty(1).uniform_(left_gamma, right_gamma))
        else:
            self.register_parameter(
                "_gamma", nn.Parameter(torch.tensor(0.8), requires_grad=False)
            )

        self._pos_w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02)
        )

    def forward(self, relative_temporal_intervals):
        # Log-scale the relative interval before the exponential-power kernel.
        #
        # NUMERICAL FIX (novel variant, NOT literally Meta's fuxi_beta form):
        # relative_temporal_intervals arrives as |t_i - t_j| in RAW UNIX SECONDS.
        # The kernel is alpha * gamma^(rel^beta) with gamma in [0.5, 0.99). For
        # any realistic delta (minutes -> days, i.e. hundreds to millions of
        # seconds), rel^beta is enormous and gamma^(that) underflows to ~0,
        # killing the entire temporal channel. FuXi-alpha / HSTU avoid this by
        # log-bucketizing the delta first. We apply the same idea via log1p so
        # the temporal bias stays numerically alive and monotone in the delta.
        scaled_intervals = torch.log1p(relative_temporal_intervals.clamp(min=0))
        ts_bias = self._alpha * torch.pow(
            self._gamma, torch.pow(scaled_intervals, self._beta)
        )

        # Build the relative-position bias at the RUNTIME sequence length N (the
        # temporal intervals are [B, N, N]), not the model's max_seq_len. Using
        # max_seq_len produced a [1, max_seq_len, max_seq_len] pos_bias that
        # mismatched the [B, N, N] ts_bias whenever N < max_seq_len (a latent
        # crash the projection-dim bug hid). Slice _pos_w to the centered
        # (2N-1)-wide window so the Toeplitz construction uses the right length.
        N = relative_temporal_intervals.size(1)
        max_r = self.max_seq_len
        # centered slice of the (2*max_r-1)-long _pos_w to length 2N-1
        center = max_r - 1
        pos_w = self._pos_w[center - (N - 1) : center + N]  # length 2N-1
        t = F.pad(pos_w, [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        pos_bias = t[:, :, r:-r]

        return ts_bias, pos_bias


class FuXiGammaBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        linear_dim,
        attention_dim,
        dropout_ratio,
        num_heads,
        max_seq_len,
        linear_activation="silu",
        ffn_multiply=1.0,
        range_alpha=0.1,
        left_beta=0.5,
        right_beta=2.0,
        gamma_learnable=True,
        left_gamma=0.5,
        right_gamma=0.99,
        epsilon=1e-6,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.num_heads = num_heads
        self.linear_activation = linear_activation
        self.epsilon = epsilon

        # Projection width MUST match the u/v split below:
        #   u : linear_dim * num_heads * 2   (gate, multiplies the aggregated v)
        #   v : linear_dim * num_heads * 1   (value, aggregated by ts/pos bias)
        # -> total = linear_dim * num_heads * 3.  The previous code sized _uv as
        # embedding_dim*3 (192 with defaults) while the split wanted
        # linear_dim*num_heads*3 (384) -> a guaranteed runtime crash on every
        # forward. The downstream norm/output are sized to the value width
        # (linear_dim*num_heads*2 = the concatenated ts+pos channels), not
        # embedding_dim*2, so hidden width is decoupled from embedding_dim.
        lin = linear_dim * num_heads
        self._lin = lin

        self._norm_input = RMSNorm(embedding_dim, eps=epsilon)
        self._norm_attn_output = RMSNorm(lin * 2, eps=epsilon)
        self._norm_ffn = RMSNorm(embedding_dim, eps=epsilon)

        self._uv = nn.Parameter(
            torch.empty(embedding_dim, lin * 3).normal_(mean=0, std=0.02)
        )

        self.rel_bias = TemporalPositionalBias(
            max_seq_len,
            range_alpha,
            left_beta,
            right_beta,
            gamma_learnable,
            left_gamma,
            right_gamma,
        )

        self._o = nn.Linear(lin * 2, embedding_dim)
        nn.init.xavier_uniform_(self._o.weight)

        self.swiglu = SwiGLU(
            embedding_dim, int(embedding_dim * ffn_multiply), dropout_ratio, bias=False
        )
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, timestamps=None, causal_mask=None):
        B, N, D = x.shape

        normed_x = self._norm_input(x)
        proj = torch.mm(normed_x.view(-1, D), self._uv).view(B, N, -1)
        if self.linear_activation == "silu":
            proj = F.silu(proj)

        u, v = torch.split(
            proj,
            [self.linear_dim * self.num_heads * 2, self.linear_dim * self.num_heads],
            dim=-1,
        )

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

        o_input = u * self._norm_attn_output(combined)
        o_output = self._o(self.dropout(o_input)) + x

        ffn_input = self._norm_ffn(o_output)
        output = self.dropout(self.swiglu(ffn_input)) + o_output

        return output


class FuXiGamma(DeepSequentialModel):
    """FuXi-γ: Exponential-Power Temporal Encoder with SwiGLU FFN."""

    def __init__(self, config: FuXiGammaConfig):
        super().__init__(config)
        self.save_hyperparameters()
        self.epsilon = config.get("epsilon", 1e-6)
        self._dropout_rate = config.get("dropout_rate", 0.5)

        # Learnable absolute position embedding (the FuXi-gamma paper notes the
        # absolute position is supplied here; the relative positional channel is
        # complementary). Truncated-normal init, scale by sqrt(d), dropout, mask.
        self.pos_embedding = nn.Embedding(config.max_seq_length, self.embedding_dim)
        truncated_normal(
            self.pos_embedding.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self.embedding_dim),
        )
        self.emb_dropout = nn.Dropout(p=self._dropout_rate)

        self.fuxi_blocks = nn.ModuleList(
            [
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
                    self.epsilon,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    config.max_seq_length, config.max_seq_length, dtype=torch.bool
                ),
                diagonal=1,
            ),
        )
        self._init_weights()

    def get_hidden_states(self, sequences, sequence_lengths):
        B, N = sequences.size()
        item_embs = self.item_embedding(sequences)
        positions = torch.arange(N, device=sequences.device).unsqueeze(0).expand(B, -1)
        # Reference input preprocessing (shared with FuXi-alpha): scale by
        # sqrt(d), add learnable absolute position, dropout, zero out padding.
        x = item_embs * (self.embedding_dim**0.5) + self.pos_embedding(positions)
        x = self.emb_dropout(x)
        valid_mask = (sequences != 0).unsqueeze(-1).float()
        x = x * valid_mask

        # Paper Eq. 7 operates on the integer relative temporal matrix
        # T_{i,j} = |t_i - t_j| cast to float32 ("Pre-Conversion of Data Type").
        # Fall back to positional deltas only when real timestamps are absent.
        timestamps = self.get_batch_timestamps()
        if timestamps is not None:
            timestamps = timestamps.to(sequences.device).float()
        else:
            timestamps = positions.float()

        # Slice the [max_seq_length, max_seq_length] causal buffer to the actual
        # sequence length N; the per-block ts/pos bias tensors are [1, N, N], so
        # the full-length mask would broadcast-mismatch (was a latent crash the
        # dimension bug masked).
        causal = self.causal_mask[:N, :N]
        for block in self.fuxi_blocks:
            x = block(x, timestamps=timestamps, causal_mask=causal)

        return x

    def predict_next(self, sequences, sequence_lengths):
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        logits = torch.matmul(
            hidden_states, self.get_output_embeddings().transpose(0, 1)
        )
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=sequences.size(1) - 1
        )
        return torch.softmax(logits[batch_indices, last_indices], dim=-1)

    def get_sequence_embedding(self, sequences, sequence_lengths):
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=hidden_states.size(1) - 1
        )
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
        # Only (re)initialize the item embedding here. Block-internal weights
        # (_uv normal(0.02), _o xavier, SwiGLU) are initialized in their own
        # __init__ and must not be clobbered — mirrors the reference, which
        # skips re-init for block submodules.
        nn.init.normal_(self.item_embedding.weight, std=0.02)
