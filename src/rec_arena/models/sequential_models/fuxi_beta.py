"""FuXi-β: dense-faithful port of the reference ``fuxi_beta`` block.

This is a NEW model, independent of FuXi-γ. It reproduces the reference
``fuxi_beta.py`` (FunctionalRelativeAttentionBias + FuXiBetaBlockJagged +
MultistageFeedforwardNeuralNetwork) with the jagged/fbgemm kernels replaced by
plain dense tensor ops. The reference jagged calls only pad/unpad the variable
length sequences; the dense equivalent is a straight reshape to [B, N, H, lin],
so there is no functional difference (padding rows carry zero embeddings and
therefore contribute nothing to the causal attention aggregation).

Faithfulness notes vs. the reference:
  * ``torch.ops.fbgemm.dense_to_jagged`` / ``jagged_to_padded_dense`` -> dense
    reshape ``[B, N, H, lin]``. Equivalent because padding tokens have zero
    embeddings (get_hidden_states zeroes them) and are never attended to by any
    valid position under the causal mask.
  * Bare ``F.rms_norm`` (no learnable affine) everywhere, matching the reference.
  * The relative-position (`_pos_w`) Toeplitz bias and the causal mask are built
    at the RUNTIME sequence length N, not ``max_seq_length`` -- this avoids the
    [1, max, max] vs [B, N, N] broadcast mismatch when N < max_seq_length.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sequential import DeepSequentialModel
from ...configs.defaults.fuxi_beta import FuXiBetaConfig


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


class FunctionalRelativeAttentionBias(nn.Module):
    """Temporal bias ``f(Δt)`` + learnable positional Toeplitz bias.

    Faithful to the reference ``FunctionalRelativeAttentionBias``. ``func_type``
    selects the closed-form (or MLP) temporal decay applied to the relative
    timestamp matrix; ``_pos_w`` provides a learnable relative-position bias.

    The positional bias is materialized at the runtime N (from the input
    timestamps), slicing ``_pos_w`` to a centered ``2N-1`` window so the
    Toeplitz construction is length-correct when N < max_seq_len.
    """

    def __init__(self, max_seq_len: int, func_type: str, func_params: dict) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self._func_type = func_type

        def _p(low_high):
            low, high = low_high
            return nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(low, high))

        if func_type == "linear":
            self._lin_a = _p(func_params["lin_a_range"])
            self._lin_b = _p(func_params["lin_b_range"])
        elif func_type == "log":
            self._log_a = _p(func_params["log_a_range"])
            self._log_b = _p(func_params["log_b_range"])
            self._log_c = _p(func_params["log_c_range"])
        elif func_type == "exp":
            self._exp_a = _p(func_params["exp_a_range"])
            self._exp_b = _p(func_params["exp_b_range"])
        elif func_type == "sin":
            self._sin_a = _p(func_params["sin_a_range"])
            self._sin_b = _p(func_params["sin_b_range"])
            self._sin_c = _p(func_params["sin_c_range"])
            self._sin_d = nn.Parameter(
                torch.full((1,), float(func_params["sin_d_init"]), dtype=torch.float32)
            )
        elif func_type == "pow":
            self._pow_a = _p(func_params["pow_a_range"])
            self._pow_b = _p(func_params["pow_b_range"])
        elif func_type == "mixed":
            self._lin_a = _p(func_params["lin_a_range"])
            self._lin_b = _p(func_params["lin_b_range"])
            self._log_a = _p(func_params["log_a_range"])
            self._log_b = _p(func_params["log_b_range"])
            self._log_c = _p(func_params["log_c_range"])
            self._exp_a = _p(func_params["exp_a_range"])
            self._exp_b = _p(func_params["exp_b_range"])
            self._sin_a = _p(func_params["sin_a_range"])
            self._sin_b = _p(func_params["sin_b_range"])
            self._sin_c = _p(func_params["sin_c_range"])
            self._sin_d = nn.Parameter(
                torch.full((1,), float(func_params["sin_d_init"]), dtype=torch.float32)
            )
            self._pow_a = _p(func_params["pow_a_range"])
            self._pow_b = _p(func_params["pow_b_range"])
        elif func_type == "nn":
            h = func_params["nn_hidden_dim"]
            self._nn_a = nn.Linear(1, h)
            self._nn_b = nn.Linear(h, h)
            self._nn_c = nn.Linear(h, 1)
        elif func_type == "zero":
            pass
        else:
            raise ValueError(f"Unknown function type {func_type}")

        self._func_map = {
            "linear": self.f_lin,
            "log": self.f_log,
            "exp": self.f_exp,
            "sin": self.f_sin,
            "pow": self.f_pow,
            "mixed": self.f_mix,
            "nn": self.f_nn,
            "zero": self.f_zero,
        }

        self._pos_w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02)
        )

    def f_lin(self, x):
        return self._lin_a * x + self._lin_b

    def f_log(self, x):
        x = torch.relu(x)
        return self._log_a * torch.log(1 + torch.relu(self._log_b) * x) + self._log_c

    def f_exp(self, x):
        x = torch.relu(x)
        exp_b = torch.exp(self._exp_b)
        return self._exp_a * torch.exp(-exp_b * x)

    def f_sin(self, x):
        return self._sin_c * torch.sin(self._sin_a * x + self._sin_b) + self._sin_d

    def f_pow(self, x):
        x = torch.relu(x) + 1
        return self._pow_a * torch.pow(x, -self._pow_b)

    def f_mix(self, x):
        return (
            self.f_lin(x)
            + self.f_log(x)
            + self.f_exp(x)
            + self.f_sin(x)
            + self.f_pow(x)
        ) / 5

    def f_nn(self, x):
        x = self._nn_a(x.to(torch.float32).unsqueeze(-1))
        x = self._nn_b(torch.sin(x))
        x = F.silu(x)
        return self._nn_c(x).squeeze(-1)

    def f_zero(self, x):
        return torch.zeros_like(x, device=x.device)

    def f(self, x):
        return self._func_map[self._func_type](x)

    def forward(self, all_timestamps: torch.Tensor):
        """Args: all_timestamps [B, N]. Returns (pos_bias [1, N, N], ts_bias [B, N, N])."""
        N = all_timestamps.size(1)

        # ---- Positional Toeplitz bias, built at runtime N. Slice _pos_w (length
        # 2*max_seq_len-1) to a centered 2N-1 window so N < max_seq_len works.
        center = self.max_seq_len - 1
        pos_w = self._pos_w[center - (N - 1) : center + N]  # length 2N-1
        t = F.pad(pos_w, [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        rel_pos_bias = t[:, :, r:-r]  # [1, N, N]

        # ---- Temporal bias. ext = [t_0..t_{N-1}, t_{N-1}] (causal), delta then f.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        delta = ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        rel_ts_bias = self.f(delta)  # [B, N, N]

        return rel_pos_bias, rel_ts_bias


class MultistageFeedforwardNeuralNetwork(nn.Module):
    """Reference MFFN: lin0(dropout(X)) + X0; rms_norm; silu(lin1)*lin3; lin2 + X.

    lin1/lin2/lin3 are bias-free; the norm is bare F.rms_norm (no affine); the
    dropout is applied to the raw input and again to the normed hidden.
    """

    def __init__(
        self,
        ams_output_size,
        input_size,
        hidden_size,
        output_size,
        dropout_ratio: float,
        bias: bool = False,
        single_stage: bool = False,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.lin0 = nn.Linear(ams_output_size, input_size)
        self.is_single_stage = single_stage
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        if not single_stage:
            self.lin1 = nn.Linear(input_size, hidden_size, bias=bias)
            self.lin2 = nn.Linear(hidden_size, output_size, bias=bias)
            self.lin3 = nn.Linear(input_size, hidden_size, bias=bias)

    def forward(self, X, X0):
        X = self.lin0(F.dropout(X, p=self.dropout_ratio, training=self.training)) + X0
        if not self.is_single_stage:
            normed_X = F.rms_norm(X, normalized_shape=[self.input_size], eps=self.eps)
            normed_X = F.dropout(
                normed_X, p=self.dropout_ratio, training=self.training
            )
            X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
            X = self.lin2(X1) + X
        return X

    def init(self):
        nn.init.xavier_uniform_(self.lin0.weight)
        if not self.is_single_stage:
            nn.init.xavier_uniform_(self.lin1.weight)
            nn.init.xavier_uniform_(self.lin2.weight)
            nn.init.xavier_uniform_(self.lin3.weight)


class FuXiBetaBlock(nn.Module):
    """Dense port of ``FuXiBetaBlockJagged``.

    Projection width = linear_dim * num_heads * 3 -> split u [lin*H*2] (gate) and
    v [lin*H] (value). v is reshaped to [B, N, H, lin] (the dense equivalent of
    the reference's jagged_to_padded_dense reshape), aggregated by the masked
    positional and temporal biases via einsum, concatenated, rms-normed, gated by
    u, then passed through the MFFN with the block input as the residual.
    """

    def __init__(
        self,
        embedding_dim: int,
        linear_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        num_heads: int,
        max_seq_len: int,
        func_type: str,
        func_params: dict,
        linear_activation: str = "silu",
        ffn_multiply: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_dim
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._linear_activation = linear_activation
        self._eps = epsilon

        lin = linear_dim * num_heads
        self._lin = lin

        # emb x (lin * H * 3): u = 2 chunks, v = 1 chunk.
        self._uvqk = nn.Parameter(
            torch.empty(embedding_dim, lin * 3).normal_(mean=0, std=0.02)
        )

        self.rel_bias = FunctionalRelativeAttentionBias(
            max_seq_len=max_seq_len,
            func_type=func_type,
            func_params=func_params,
        )

        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size=lin * 2,
            input_size=embedding_dim,
            hidden_size=int(embedding_dim * ffn_multiply),
            output_size=embedding_dim,
            dropout_ratio=dropout_ratio,
            single_stage=False,
            epsilon=epsilon,
        )
        self._mffn.init()

    def _norm_input(self, x):
        return F.rms_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x):
        return F.rms_norm(
            x, normalized_shape=[self._lin * 2], eps=self._eps
        )

    def forward(self, x, timestamps, causal_mask=None):
        B, N, D = x.shape

        normed_x = self._norm_input(x)
        proj = torch.mm(normed_x.reshape(-1, D), self._uvqk).view(B, N, -1)
        if self._linear_activation == "silu":
            proj = F.silu(proj)
        # "none" leaves proj unchanged (reference parity).

        u, v = torch.split(
            proj,
            [self._linear_dim * self._num_heads * 2, self._linear_dim * self._num_heads],
            dim=-1,
        )

        pos_attn, ts_attn = self.rel_bias(timestamps)

        if causal_mask is not None:
            # invalid_attn_mask = 1 where allowed (causal_mask True == future).
            mask = (~causal_mask).to(pos_attn.dtype)
            pos_attn = pos_attn * mask.unsqueeze(0)
            ts_attn = ts_attn * mask.unsqueeze(0)

        # Dense equivalent of jagged_to_padded_dense reshape: [B, N, H, lin].
        v_reshaped = v.reshape(B, N, self._num_heads, self._linear_dim)

        output_pos = torch.einsum("bnm,bmhd->bnhd", pos_attn, v_reshaped)
        output_ts = torch.einsum("bnm,bmhd->bnhd", ts_attn, v_reshaped)

        combined = torch.concat([output_pos, output_ts], dim=-1).reshape(
            B, N, self._num_heads * self._linear_dim * 2
        )

        attn_output = u * self._norm_attn_output(combined)

        return self._mffn(attn_output, x)


class FuXiBeta(DeepSequentialModel):
    """FuXi-β: Functional relative attention bias with a multistage FFN.

    Dense-faithful port of the reference ``fuxi_beta`` (no fbgemm/jagged kernels).
    """

    def __init__(self, config: FuXiBetaConfig):
        super().__init__(config)
        self.save_hyperparameters()
        self.epsilon = config.get("epsilon", 1e-6)
        self._dropout_rate = config.get("dropout_rate", 0.2)
        self._attention_dim = config.attention_dim
        self._func_type = config.func_type

        # Learnable absolute position embedding (shared FuXi-family input
        # preprocessing): scale item emb by sqrt(d), add position, dropout, mask.
        self.pos_embedding = nn.Embedding(config.max_seq_length, self.embedding_dim)
        truncated_normal(
            self.pos_embedding.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self.embedding_dim),
        )
        self.emb_dropout = nn.Dropout(p=self._dropout_rate)

        # Collect all temporal-bias init ranges from config; the active func_type
        # consumes the relevant subset (others are inert but still config-driven).
        func_params = {
            "lin_a_range": config.lin_a_range,
            "lin_b_range": config.lin_b_range,
            "log_a_range": config.log_a_range,
            "log_b_range": config.log_b_range,
            "log_c_range": config.log_c_range,
            "exp_a_range": config.exp_a_range,
            "exp_b_range": config.exp_b_range,
            "sin_a_range": config.sin_a_range,
            "sin_b_range": config.sin_b_range,
            "sin_c_range": config.sin_c_range,
            "sin_d_init": config.sin_d_init,
            "pow_a_range": config.pow_a_range,
            "pow_b_range": config.pow_b_range,
            "nn_hidden_dim": config.nn_hidden_dim,
        }

        self.fuxi_blocks = nn.ModuleList(
            [
                FuXiBetaBlock(
                    embedding_dim=self.embedding_dim,
                    linear_dim=config.linear_dim,
                    attention_dim=config.attention_dim,
                    dropout_ratio=config.dropout_rate,
                    num_heads=config.num_heads,
                    max_seq_len=config.max_seq_length,
                    func_type=config.func_type,
                    func_params=func_params,
                    linear_activation=config.get("linear_activation", "silu"),
                    ffn_multiply=config.ffn_multiply,
                    epsilon=self.epsilon,
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
        x = item_embs * (self.embedding_dim**0.5) + self.pos_embedding(positions)
        x = self.emb_dropout(x)
        # Zero padding rows: dense-faithful equivalent of jagged excluding pads.
        # Padding tokens then contribute nothing to the causal aggregation.
        valid_mask = (sequences != 0).unsqueeze(-1).float()
        x = x * valid_mask

        # Real interaction timestamps drive the temporal bias; fall back to
        # positional deltas when timestamps are unavailable.
        timestamps = self.get_batch_timestamps()
        if timestamps is not None:
            timestamps = timestamps.to(sequences.device).float()
        else:
            timestamps = positions.float()

        # Slice the causal buffer to runtime N (the ts/pos bias are [., N, N]).
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
        # Only (re)initialize the item embedding here; block-internal weights
        # (_uvqk normal(0.02), MFFN xavier, temporal-bias params) are initialized
        # in their own __init__ and must not be clobbered -- mirrors the
        # reference, which skips re-init for block submodules.
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()  # keep PAD row at 0
