"""FuXi-Linear: dense-faithful port of the official FuXi-Linear model.

This is a DENSE re-implementation (no fbgemm / no jagged / no chunkwise
recurrence) of the reference ``FuXiLinearBlockJagged`` and its three parallel
attention channels:

  * Retention          -- linear-attention "latent" channel with per-head
                          exponential decay (RetNet-style).
  * LinearTemporalChannel  -- kernelized relative *time* bias (KRAB) using
                          sinusoidal timestamp features + per-head learned decay.
  * LinearPositionalChannel -- learned absolute-position sinusoidal attention.

Faithfulness note (dense vs. chunkwise equivalence)
---------------------------------------------------
The reference computes linear attention chunk-recurrently
(``fuxi_modules/linear_attn.py:chunkwise_forward`` + a custom autograd
``ChunkwiseHiddenStateFunction`` in ``linear_attn_fn.py``). That chunked form is
mathematically identical to the plain quadratic linear-attention-with-decay:

    y[p] = sum_{m<=p} (q_p . k_m) * exp(-D(m, p)) * v_m

where the decay ``D(m, p) = sum_{j=m}^{p} log_decay_step[j]`` is the cumulative
per-step log-decay between positions m and p (inclusive), realized as
``clamp(cumsum[p+1] - cumsum[m], min=0)`` with ``cumsum`` over ``[0, log_decay]``.
This dense form was numerically verified against the reference
``chunkwise_forward`` (which uses the pure-torch autograd recurrence) to
``max|diff| ~ 1e-6`` -- see the module ``__main__`` self-check. It avoids both
fbgemm and the chunk machinery while remaining numerically equivalent.

Dimensional constraint
----------------------
The reference layer-norms each channel output over ``embedding_dim`` and feeds
the channels the layer-normed residual stream (``embedding_dim`` wide) through a
per-channel value projection sized ``value_dim = linear_dim * num_heads``. This
implicitly requires ``linear_dim * num_heads == embedding_dim`` (validated in the
config). The temporal channel further splits the value into ``2 * num_heads``
heads, requiring ``embedding_dim % (2 * channel_t_num_heads) == 0``.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sequential import DeepSequentialModel
from ...configs.defaults.fuxi_linear import FuXiLinearConfig


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


def _apply_rope(inputs: torch.Tensor, sin_bases: torch.Tensor, cos_bases: torch.Tensor):
    """RoPE-style rotation used by the reference Retention (dense port).

    Mirrors ``apply_multiplication_with_lambda``: splits the last dim into two
    halves and applies a 2D rotation.
    inputs: (B, N, h, d); sin/cos bases: (N, h, d/2) broadcast over B.
    """
    dim = inputs.shape[-1]
    assert dim % 2 == 0
    half_dim = dim // 2
    chk0, chk1 = inputs.split([half_dim, half_dim], dim=-1)
    if sin_bases.dim() == 3:
        sin_bases = sin_bases.unsqueeze(0)
        cos_bases = cos_bases.unsqueeze(0)
    pos_real = chk0 * cos_bases - chk1 * sin_bases
    pos_img = chk0 * sin_bases + chk1 * cos_bases
    return torch.concat([pos_real, pos_img], dim=-1)


class DenseRetention(nn.Module):
    """Dense quadratic-with-decay port of the reference ``Retention`` module.

    Realizes ``y[p] = sum_{m<=p} (q_p . k_m) * gamma_h^(clamp(p-m,0)+1) * v_m``
    per head, which is the non-chunked equivalent of the reference
    ``chunkwise_forward`` when the per-step log-decay is the constant per-head
    ``gamma`` (verified numerically to ~1e-6).
    """

    def __init__(self, head_dim: int, num_heads: int, use_rope: bool = False):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._use_rope = use_rope
        # Per-head decay rate. Reference: gamma = exp(-cumsum(softplus(param))),
        # giving a monotonically decreasing set of decays in (0, 1).
        self._gamma = nn.Parameter(torch.empty(num_heads).normal_(mean=0, std=0.02))
        if use_rope:
            half_head_dim = head_dim // 2
            theta = torch.exp(
                -math.log(10000) * torch.arange(half_head_dim) / max(half_head_dim, 1)
            )
            # Non-learnable, matching the reference (requires_grad=False).
            self._lambda = nn.Parameter(theta, requires_grad=False)

    def _get_gamma(self) -> torch.Tensor:
        gamma = F.softplus(self._gamma)
        gamma = torch.cumsum(gamma, dim=0)
        return torch.exp(-gamma)

    def forward(
        self,
        q: torch.Tensor,  # (B, N, h, d_attn)
        k: torch.Tensor,  # (B, N, h, d_attn)
        v: torch.Tensor,  # (B, N, h, d_v)
        causal: torch.Tensor,  # (N, N) lower-triangular ones (incl. diagonal)
    ) -> torch.Tensor:
        B, N = q.shape[0], q.shape[1]

        if self._use_rope:
            position = torch.arange(N, dtype=q.dtype, device=q.device)
            bases = torch.einsum("n,d->nd", position, self._lambda).unsqueeze(-2)
            sin_bases, cos_bases = torch.sin(bases), torch.cos(bases)
            q = _apply_rope(q, sin_bases, cos_bases)
            k = _apply_rope(k, sin_bases, cos_bases)

        gamma = self._get_gamma()  # (h,)
        pos = torch.arange(N, device=q.device, dtype=torch.float32)
        # diff[p, m] = clamp(p - m, 0) + 1  (shift=1 in the reference).
        diff = torch.clamp(pos.unsqueeze(1) - pos.unsqueeze(0), min=0) + 1.0
        decay = torch.exp(-torch.einsum("nm,h->hnm", diff, gamma))  # (h, N, N)

        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k)  # (B, h, N, N)
        qk_attn = qk_attn * decay.unsqueeze(0) * causal
        out = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v)  # (B, N, h, d_v)
        return out.reshape(B, N, -1)


class DenseLinearTemporalChannel(nn.Module):
    """Dense port of the reference ``LinearTemporalChannel`` (KRAB).

    Query/key are sinusoidal features of the interaction *timestamps* (so they
    carry no dependence on the input embeddings, preserving causality), and the
    decay is a per-head learned function of the relative time interval. The
    aggregation is masked causally, so the output at position p depends only on
    values (input embeddings) at positions m <= p.
    """

    def __init__(
        self,
        linear_dim: int,  # == value_dim == embedding_dim
        num_heads: int,
        base: float = 2.0,
        start_index: int = 0,
        base_stride: int = 1,
        use_proj: bool = True,
        learnable_gamma: bool = False,
        no_temporal_qk: bool = False,
        use_augment_connection: bool = False,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._linear_dim = linear_dim
        self._use_proj = use_proj
        self._no_temporal_qk = no_temporal_qk
        self._use_augment_connection = use_augment_connection
        self._pi2 = math.pi * 2

        exps = torch.arange(
            start_index, start_index + num_heads * base_stride, base_stride
        )
        # Integer periods for the timestamp modulo (kept as long for exact
        # integer arithmetic on Unix-second timestamps).
        self.register_buffer("_intervals", torch.pow(base, exps.to(torch.float64)).long())
        self.register_buffer(
            "_scale_factor",
            self._pi2 * torch.pow(1.0 / base, exps.to(torch.float32)),
        )

        if learnable_gamma:
            self._gamma = nn.Parameter(
                torch.empty(num_heads, dtype=torch.float32).normal_(std=0.02)
            )
        else:
            self.register_buffer("_gamma", torch.zeros(1, dtype=torch.float32))

        if use_proj:
            self.proj_v = nn.Linear(linear_dim, linear_dim, bias=False)

        if use_augment_connection:
            self.alpha = nn.Parameter(
                torch.empty(num_heads * 2).normal_(mean=0, std=0.02)
            )
            self.beta = nn.Parameter(torch.empty(num_heads * 2).fill_(1.0))

    def _get_decay(self, diff: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        hdiff = torch.einsum("...,h->...h", diff, self._scale_factor)
        return torch.pow(gamma, hdiff)

    def _get_query_key(self, all_timestamps: torch.Tensor, gamma: torch.Tensor):
        """Sinusoidal timestamp features. Returns q, k of shape (B, N, 2h, 2)."""
        B, N = all_timestamps.shape
        h = self._intervals.shape[-1]
        if self._no_temporal_qk:
            q = torch.ones(
                B, N, h * 2, 2, dtype=torch.float32, device=all_timestamps.device
            ) / math.sqrt(2)
            return q, q

        ts_long = all_timestamps.long()
        theta_t = (
            torch.remainder(ts_long[:, :, None], self._intervals[None, None, :]).to(
                torch.float32
            )
            * self._scale_factor[None, None, :]
        )  # (B, N, h)
        cos_t, sin_t = torch.cos(theta_t), torch.sin(theta_t)
        # (B, N, 2h, 2)
        k = torch.stack([cos_t, sin_t], dim=3).repeat(1, 1, 2, 1)

        # q is shifted by one position (uses the *next* timestamp), matching the
        # reference. This is causal-safe: q/k depend only on timestamps (fixed),
        # never on input embeddings, and aggregation is causally masked.
        q_sin = torch.stack([sin_t[:, 1:], -cos_t[:, 1:]], dim=3)
        q_cos = torch.stack([cos_t[:, 1:], sin_t[:, 1:]], dim=3)
        q = torch.concat([q_sin, q_cos], dim=2)  # (B, N-1, 2h, 2)
        q = torch.concat([q, q[:, -2:-1]], dim=1)  # (B, N, 2h, 2) -- faithful quirk
        return q, k

    def forward(
        self,
        v_input: torch.Tensor,  # normed residual stream (B, N, embedding_dim)
        timestamps: torch.Tensor,  # (B, N) float/int timestamps
        causal: torch.Tensor,  # (N, N) lower-tri ones
    ) -> torch.Tensor:
        B, N = timestamps.shape
        gamma = torch.sigmoid(self._gamma)

        v = self.proj_v(v_input) if self._use_proj else v_input
        x = v.reshape(B, N, 2 * self._num_heads, -1)  # (B, N, 2h, head_dim)

        interval = torch.clamp(timestamps[:, 1:] - timestamps[:, :-1], min=0).float()
        interval = F.pad(interval, (0, 1))  # (B, N)
        hinterval = torch.einsum("h,bn->bnh", self._scale_factor, interval)  # (B,N,h)

        q, k = self._get_query_key(timestamps, gamma)  # (B, N, 2h, 2)

        log_gamma = torch.log(gamma)
        # padded_log_decay == 1 (reference passes ones), so it drops out.
        log_decay_pos = (hinterval * -log_gamma[None, None, :]).repeat(1, 1, 2)  # (B,N,2h)

        # Cumulative per-step decay -> (B, 2h, N, N); D(m,p)=cumsum[p+1]-cumsum[m].
        ext = torch.cat([torch.zeros_like(log_decay_pos[:, :1]), log_decay_pos], dim=1)
        cs = torch.cumsum(ext, dim=1).permute(0, 2, 1)  # (B, 2h, N+1)
        ldm = torch.clamp(cs[:, :, 1:, None] - cs[:, :, None, :-1], min=0)
        decay_map = torch.exp(-ldm) * causal.unsqueeze(0).unsqueeze(0)  # (B, 2h, N, N)

        attn = torch.einsum("bnhd,bmhd->bhnm", q, k) * decay_map
        y = torch.einsum("bhnm,bmhd->bnhd", attn, x)  # (B, N, 2h, head_dim)

        if self._use_augment_connection:
            y = torch.einsum("bnhd,h->bnhd", y, self.alpha) + torch.einsum(
                "bnhd,h->bnhd", x, self.beta
            )
        return y.reshape(B, N, -1)


class DenseLinearPositionalChannel(nn.Module):
    """Dense port of the reference ``LinearPositionalChannel``.

    Learns an absolute-position sinusoidal embedding table and attends via
    ``<emb_i, emb_j>`` (causally masked). The table is sliced to the *runtime*
    sequence length N (never max_seq_length) to avoid the latent shape crash.
    """

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int = 32,
        aug_current: bool = True,
        use_proj: bool = True,
        value_dim: Optional[int] = None,
        sinusoidal_base: float = 10000.0,
    ):
        super().__init__()
        assert embedding_dim % 2 == 0, "positional channel dim must be even"
        self._use_proj = use_proj
        self._aug_current = aug_current

        half_dim = embedding_dim // 2
        theta = torch.exp(
            -math.log(sinusoidal_base) * torch.arange(half_dim) / half_dim
        )
        bases = torch.arange(max_seq_len)[:, None] * theta[None, :]
        emb_weight = torch.concat([torch.sin(bases), torch.cos(bases)], dim=1)
        self._emb = nn.Parameter(emb_weight, requires_grad=True)  # (max_seq_len, dim)

        if use_proj:
            assert value_dim is not None
            self.proj_p = nn.Linear(value_dim, value_dim, bias=False)

        if aug_current:
            self._alpha = nn.Parameter(
                torch.empty(1, dtype=torch.float32).normal_(std=0.02)
            )
            self._beta = nn.Parameter(torch.empty(1, dtype=torch.float32).fill_(1.0))

    def forward(
        self,
        v_input: torch.Tensor,  # normed residual stream (B, N, embedding_dim)
        causal: torch.Tensor,  # (N, N) lower-tri ones
    ) -> torch.Tensor:
        N = v_input.size(1)
        v = self.proj_p(v_input) if self._use_proj else v_input  # (B, N, value_dim)

        emb = self._emb[:N]  # RUNTIME slice -> (N, dim)
        attn_weights = (
            torch.einsum("nd,md->nm", emb, emb) / (emb.shape[-1] // 2) * causal
        )  # (N, N)
        y = torch.einsum("nm,bmd->bnd", attn_weights, v)

        if self._aug_current:
            y = y * self._alpha + v * self._beta
        return y


class DenseMFFN(nn.Module):
    """Dense port of the reference ``MultistageFeedforwardNeuralNetwork``."""

    def __init__(
        self,
        ams_output_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_ratio: float,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.lin0 = nn.Linear(ams_output_size, input_size, bias=False)
        self.lin1 = nn.Linear(input_size, hidden_size, bias=False)
        self.lin2 = nn.Linear(hidden_size, output_size, bias=False)
        self.lin3 = nn.Linear(input_size, hidden_size, bias=False)
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        self.init()

    def init(self):
        with torch.no_grad():
            for lin in (self.lin0, self.lin1, self.lin2, self.lin3):
                lin.weight.normal_(mean=0, std=0.02)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        x = self.lin0(F.dropout(x, p=self.dropout_ratio, training=self.training)) + x0
        normed_x = F.rms_norm(x, normalized_shape=[self.input_size], eps=self.eps)
        normed_x = F.dropout(normed_x, p=self.dropout_ratio, training=self.training)
        x1 = F.silu(self.lin1(normed_x)) * self.lin3(normed_x)
        x = self.lin2(x1) + x
        return x


class FuXiLinearBlock(nn.Module):
    """Dense port of ``FuXiLinearBlockJagged``.

    Splits a single projection into (u, q, k, v), runs three parallel channels
    (retention / temporal / positional), layer-norms each channel output over
    ``embedding_dim``, gates the concatenation by ``u``, and applies the
    multi-stage FFN.
    """

    def __init__(
        self,
        embedding_dim: int,
        linear_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout_ratio: float,
        max_seq_len: int,
        ffn_multiply: float = 1.0,
        linear_activation: str = "silu",
        epsilon: float = 1e-6,
        use_rope: bool = False,
        enable_temporal_channel: bool = True,
        enable_positional_channel: bool = True,
        channel_t_num_heads: int = None,
        channel_t_base: float = 2.0,
        channel_t_start_index: int = 0,
        channel_t_base_stride: int = 1,
        channel_t_use_proj: bool = True,
        channel_t_learnable_gamma: bool = False,
        channel_t_no_temporal_qk: bool = False,
        channel_t_aug_current: bool = False,
        channel_p_dim: int = 32,
        channel_p_aug_current: bool = True,
        channel_p_use_proj: bool = True,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_dim
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._linear_activation = linear_activation
        self._eps = epsilon

        query_dim = attention_dim * num_heads
        value_dim = linear_dim * num_heads
        self._value_dim = value_dim
        n_channels = 1 + int(enable_temporal_channel) + int(enable_positional_channel)
        self._attn_dim = value_dim * n_channels

        # Projection order matches the reference split: u, q, k, v.
        self._uvqk = nn.Parameter(
            torch.empty(
                embedding_dim, self._attn_dim + query_dim + query_dim + value_dim
            ).normal_(mean=0, std=0.02)
        )

        self._retention = DenseRetention(attention_dim, num_heads, use_rope=use_rope)

        self._channel_t = None
        if enable_temporal_channel:
            self._channel_t = DenseLinearTemporalChannel(
                linear_dim=value_dim,
                num_heads=channel_t_num_heads or num_heads,
                base=channel_t_base,
                start_index=channel_t_start_index,
                base_stride=channel_t_base_stride,
                use_proj=channel_t_use_proj,
                learnable_gamma=channel_t_learnable_gamma,
                no_temporal_qk=channel_t_no_temporal_qk,
                use_augment_connection=channel_t_aug_current,
            )

        self._channel_p = None
        if enable_positional_channel:
            self._channel_p = DenseLinearPositionalChannel(
                max_seq_len=max_seq_len,
                embedding_dim=channel_p_dim,
                aug_current=channel_p_aug_current,
                use_proj=channel_p_use_proj,
                value_dim=value_dim,
            )

        self._mffn = DenseMFFN(
            ams_output_size=self._attn_dim,
            input_size=embedding_dim,
            hidden_size=int(embedding_dim * ffn_multiply),
            output_size=embedding_dim,
            dropout_ratio=dropout_ratio,
            epsilon=epsilon,
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, E)
        timestamps: torch.Tensor,  # (B, N)
        causal: torch.Tensor,  # (N, N) lower-tri ones
    ) -> torch.Tensor:
        B, N, D = x.shape

        normed_x = self._norm(x)
        proj = torch.mm(normed_x.reshape(-1, D), self._uvqk).view(B, N, -1)
        if self._linear_activation == "silu":
            proj = F.silu(proj)

        u, q, k, v = torch.split(
            proj,
            [
                self._attn_dim,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._value_dim,
            ],
            dim=-1,
        )
        q = q.view(B, N, self._num_heads, self._attention_dim)
        k = k.view(B, N, self._num_heads, self._attention_dim)
        v = v.view(B, N, self._num_heads, self._linear_dim)

        outputs = []
        latent = self._retention(q, k, v, causal).reshape(B, N, self._value_dim)
        outputs.append(self._norm(latent))

        if self._channel_t is not None:
            out_t = self._channel_t(normed_x, timestamps, causal)
            outputs.append(self._norm(out_t))

        if self._channel_p is not None:
            out_p = self._channel_p(normed_x, causal)
            outputs.append(self._norm(out_p))

        combined = torch.cat(outputs, dim=-1)  # (B, N, _attn_dim)
        attn_output = u * combined
        return self._mffn(attn_output, x)


class FuXiLinear(DeepSequentialModel):
    """FuXi-Linear: chunkwise linear-attention retention encoder (dense port)."""

    def __init__(self, config: FuXiLinearConfig):
        super().__init__(config)
        self.save_hyperparameters()
        self.epsilon = config.get("epsilon", 1e-6)
        self._dropout_rate = config.get("dropout_rate", 0.2)

        # Learnable absolute position embedding on the residual stream (mirrors
        # the reference LearnablePositionalEmbedding input preprocessor and the
        # other RecArena FuXi ports). Complementary to the positional channel.
        self.pos_embedding = nn.Embedding(config.max_seq_length, self.embedding_dim)
        truncated_normal(
            self.pos_embedding.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self.embedding_dim),
        )
        self.emb_dropout = nn.Dropout(p=self._dropout_rate)

        channel_t_num_heads = getattr(config, "channel_t_num_heads", None) or config.num_heads

        self.fuxi_blocks = nn.ModuleList(
            [
                FuXiLinearBlock(
                    embedding_dim=self.embedding_dim,
                    linear_dim=config.linear_dim,
                    attention_dim=config.attention_dim,
                    num_heads=config.num_heads,
                    dropout_ratio=config.dropout_rate,
                    max_seq_len=config.max_seq_length,
                    ffn_multiply=config.ffn_multiply,
                    linear_activation=getattr(config, "linear_activation", "silu"),
                    epsilon=self.epsilon,
                    use_rope=getattr(config, "use_rope", False),
                    enable_temporal_channel=getattr(
                        config, "enable_temporal_channel", True
                    ),
                    enable_positional_channel=getattr(
                        config, "enable_positional_channel", True
                    ),
                    channel_t_num_heads=channel_t_num_heads,
                    channel_t_base=getattr(config, "channel_t_base", 2.0),
                    channel_t_start_index=getattr(config, "channel_t_start_index", 0),
                    channel_t_base_stride=getattr(config, "channel_t_base_stride", 1),
                    channel_t_use_proj=getattr(config, "channel_t_use_proj", True),
                    channel_t_learnable_gamma=(
                        getattr(config, "channel_t_learnable_gamma", False)
                        and _ == 0
                    ),
                    channel_t_no_temporal_qk=getattr(
                        config, "channel_t_no_temporal_qk", False
                    ),
                    channel_t_aug_current=getattr(config, "channel_t_aug_current", False),
                    channel_p_dim=getattr(config, "channel_p_dim", 32),
                    channel_p_aug_current=getattr(config, "channel_p_aug_current", True),
                    channel_p_use_proj=getattr(config, "channel_p_use_proj", True),
                )
                for _ in range(config.num_layers)
            ]
        )

        # Causal mask buffer sized at max_seq_length; SLICED to runtime N in
        # get_hidden_states so N < max_seq_length never mismatches (the latent
        # crash the fuxi_gamma projection-dim bug hid).
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
        x = item_embs * (self.embedding_dim**0.5) + self.pos_embedding(positions)
        x = self.emb_dropout(x)
        valid_mask = (sequences != 0).unsqueeze(-1).float()
        x = x * valid_mask

        timestamps = self.get_batch_timestamps()
        if timestamps is not None:
            timestamps = timestamps.to(sequences.device).float()
        else:
            timestamps = positions.float()

        # Lower-triangular ones (incl. diagonal) at RUNTIME N -- the reference
        # invalid_attn_mask = 1 - triu(diag=1). Built at N (not max_seq_length).
        causal = (~self.causal_mask[:N, :N]).float()

        for block in self.fuxi_blocks:
            x = block(x, timestamps=timestamps, causal=causal)

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
        # (_uvqk normal(0.02), MFFN normal(0.02), channel params) are set in
        # their own __init__ and must not be clobbered -- mirrors the reference,
        # which skips re-init for the fuxi submodules.
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()


# --------------------------------------------------------------------------- #
# Dense-vs-chunkwise numerical equivalence self-check (run directly).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/tmp/fuxi-linear")
    from generative_recommenders.modeling.sequential.fuxi_modules import linear_attn

    torch.manual_seed(0)
    B, n, h, dqk, dv = 2, 12, 3, 4, 5
    chunk_size = 4
    q = torch.randn(B, n, h, dqk)
    k = torch.randn(B, n, h, dqk)
    v = torch.randn(B, n, h, dv)

    # (1) General per-step positive log-decay: dense == reference chunkwise.
    log_decay = F.softplus(torch.randn(B, n, h))
    ref = linear_attn.chunkwise_forward(q, k, v, log_decay, chunk_size)
    ext = torch.cat([torch.zeros_like(log_decay[:, :1]), log_decay], dim=1)
    pre = torch.cumsum(ext, dim=1).permute(0, 2, 1)
    ldm = torch.clamp(pre[:, :, 1:, None] - pre[:, :, None, :-1], min=0)
    tri = torch.tril(torch.ones(n, n))
    attn = torch.einsum("bnhd,bmhd->bhnm", q, k) * torch.exp(-ldm) * tri
    dense = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(B, n, h * dv)
    print("max|diff| general log-decay :", (ref - dense).abs().max().item())

    # (2) Constant per-head gamma (the DenseRetention case) == chunkwise.
    ret = DenseRetention(dv, h)
    g = ret._get_gamma().detach()
    log_decay_c = g[None, None, :].expand(B, n, h).contiguous()
    ref2 = linear_attn.chunkwise_forward(q, k, v, log_decay_c, chunk_size)
    out = ret.forward(q, k, v, tri)  # note dv used as head_dim here for the check
    # rebuild expected with matching q/k dims
    pos = torch.arange(n)
    diff = torch.clamp(pos[:, None] - pos[None, :], min=0) + 1
    dm = torch.exp(-torch.einsum("nm,h->hnm", diff.float(), g))
    attn2 = torch.einsum("bnhd,bmhd->bhnm", q, k) * dm[None] * tri
    exp2 = torch.einsum("bhnm,bmhd->bnhd", attn2, v).reshape(B, n, h * dv)
    print("max|diff| retention const  :", (ref2 - exp2).abs().max().item())
    print("DenseRetention matches formula:", torch.allclose(out, exp2, atol=1e-5))
