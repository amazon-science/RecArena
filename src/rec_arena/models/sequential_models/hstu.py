import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..sequential import DeepSequentialModel
from ...configs.defaults.hstu import HSTUConfig


def truncated_normal(
    x: torch.Tensor, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """In-place truncated normal init (mirrors the HSTU reference util)."""
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
    return x


class RelativeBucketedTimeAndPositionBasedBias(nn.Module):
    """Relative position bias + bucketed relative-time bias (HSTU reference).

    Faithful dense version of Meta's `RelativeBucketedTimeAndPositionBasedBias`:
    the position bias is a Toeplitz construction over a learnable weight vector;
    the time bias buckets pairwise (t_j - t_i) via log-bucketization and looks up
    a learnable per-bucket weight. Returns [B, N, N].
    """

    def __init__(self, max_seq_len: int, num_buckets: int = 128):
        super().__init__()
        self.max_seq_len = max_seq_len
        self._num_buckets = num_buckets
        self._ts_w = nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02)
        )
        self._pos_w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02)
        )

    def _bucketize(self, deltas: torch.Tensor) -> torch.Tensor:
        # log-bucketization (T5/HSTU style); 0.301 ~= log10(2)
        return (torch.log(torch.abs(deltas).clamp(min=1)) / 0.301).long()

    def _position_bias(self) -> torch.Tensor:
        n = self.max_seq_len
        t = F.pad(self._pos_w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[:, :, r:-r]  # [1, N, N]

    def position_only(self, seq_len: int) -> torch.Tensor:
        """Relative-position component only [1, N, N] (timestamp-free fallback)."""
        return self._position_bias()

    def forward(self, all_timestamps: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Args: all_timestamps [B, N]. Returns [B, N, N] = pos_bias + ts_bias."""
        n = self.max_seq_len
        pos_bias = self._position_bias()  # [1, N, N]

        B = all_timestamps.size(0)
        ext = torch.cat([all_timestamps, all_timestamps[:, n - 1 : n]], dim=1)
        bucketed = torch.clamp(
            self._bucketize(ext[:, 1:].unsqueeze(2) - ext[:, :-1].unsqueeze(1)),
            min=0,
            max=self._num_buckets,
        ).detach()
        ts_bias = torch.index_select(self._ts_w, dim=0, index=bucketed.view(-1)).view(
            B, n, n
        )
        return pos_bias + ts_bias


def _hstu_attention(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Dense equivalent of the reference `_hstu_attention_maybe_from_cache`.

    Args:
        q, k: [B, N, attention_dim * num_heads]
        v:    [B, N, linear_dim * num_heads]
        rel_bias: [*, N, N] relative time+position bias (broadcasts over heads)
        invalid_attn_mask: [N, N] float in {0, 1}; 1 = keep (lower-triangular).
    Returns:
        [B, N, linear_dim * num_heads]
    """
    B, N, _ = q.shape
    qh = q.view(B, N, num_heads, attention_dim)
    kh = k.view(B, N, num_heads, attention_dim)
    vh = v.view(B, N, num_heads, linear_dim)

    # qk_attn: [B, H, N, N]
    qk_attn = torch.einsum("bnhd,bmhd->bhnm", qh, kh)
    qk_attn = qk_attn + rel_bias.unsqueeze(1)  # broadcast bias over heads
    qk_attn = F.silu(qk_attn) / N
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)

    attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, vh).reshape(
        B, N, num_heads * linear_dim
    )
    return attn_output


class SequentialTransductionUnit(nn.Module):
    """Dense equivalent of the reference `SequentialTransductionUnitJagged`.

    forward path (normalization == "rel_bias"):
        normed_x = RMS/LayerNorm(x)            # reference uses LayerNorm
        u,v,q,k  = split(SiLU(normed_x @ uvqk))
        a        = HSTU-attention(q,k,v, rel_bias, mask)
        o_input  = u * norm_attn(a)            # concat_ua=False (default)
                   or cat([u, norm_attn(a), u*norm_attn(a)])  # concat_ua=True
        out      = dropout(o_input) @ W_o + x
    """

    def __init__(
        self,
        embedding_dim: int,
        linear_dim: int,
        attention_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        concat_ua: bool = False,
        use_time_bias: bool = True,
        num_time_buckets: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_dim
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._dropout_ratio = dropout
        self._concat_ua = concat_ua
        self._use_time_bias = use_time_bias
        self._eps = eps

        # uvqk: u,v -> linear_dim*H ; q,k -> attention_dim*H  (reference sizing)
        self._uvqk = nn.Parameter(
            torch.empty(
                embedding_dim,
                linear_dim * num_heads * 2 + attention_dim * num_heads * 2,
            ).normal_(mean=0, std=0.02)
        )
        self._o = nn.Linear(
            linear_dim * num_heads * (3 if concat_ua else 1),
            embedding_dim,
        )
        nn.init.xavier_uniform_(self._o.weight)
        nn.init.zeros_(self._o.bias)

        if use_time_bias:
            self._rel_attn_bias = RelativeBucketedTimeAndPositionBasedBias(
                max_seq_len, num_time_buckets
            )
        else:
            self._rel_attn_bias = RelativeBucketedTimeAndPositionBasedBias(
                max_seq_len, num_time_buckets
            )

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def _bias(self, N: int, timestamps: Optional[torch.Tensor]) -> torch.Tensor:
        if self._use_time_bias and timestamps is not None:
            return self._rel_attn_bias(timestamps, N)
        return self._rel_attn_bias.position_only(N)

    def forward(
        self,
        x: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        normed_x = self._norm_input(x)
        proj = F.silu(torch.mm(normed_x.reshape(-1, self._embedding_dim), self._uvqk))
        proj = proj.view(B, N, -1)
        u, v, q, k = torch.split(
            proj,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=-1,
        )

        rel_bias = self._bias(N, timestamps)
        attn_output = _hstu_attention(
            num_heads=self._num_heads,
            attention_dim=self._attention_dim,
            linear_dim=self._linear_dim,
            q=q,
            k=k,
            v=v,
            rel_bias=rel_bias,
            invalid_attn_mask=invalid_attn_mask,
        )

        a = self._norm_attn_output(attn_output)
        if self._concat_ua:
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * a

        out = (
            self._o(F.dropout(o_input, p=self._dropout_ratio, training=self.training))
            + x
        )
        return out


class HSTU(DeepSequentialModel):
    """HSTU: Hierarchical Sequential Transduction Unit.

    Dense-equivalent of Meta's reference implementation
    (`generative_recommenders/.../sequential/hstu.py`,
    normalization="rel_bias", linear_config="uvqk"):
      - SiLU over the full u/v/q/k projection, attention normalized by 1/n
      - relative time + position bias from real timestamps
      - decoupled attention_dim (dqk) and linear_dim (dv)
      - gated transform `u * norm(attn)` (concat_ua=False) or `cat([u,a,u*a])`
      - bias-free uvqk; residual connection
      - input preprocessing: emb * sqrt(d) + abs. position, dropout, pad mask

    Paper: "Actions Speak Louder than Words" (ICML 2024),
    https://arxiv.org/abs/2402.17152
    """

    def __init__(self, config: HSTUConfig):
        super().__init__(config)
        self.save_hyperparameters()

        self._dropout_rate = config.get("dropout_rate", 0.2)
        self.use_time_bias = bool(config.get("use_time_bias", True))
        self.concat_ua = bool(config.get("concat_ua", False))
        num_time_buckets = int(config.get("num_time_buckets", 128))
        # Decoupled per-head dims (reference dqk / dv). Default to head_dim.
        head_dim = self.embedding_dim // config.num_heads
        attn_dim = config.get("attention_dim", None)
        lin_dim = config.get("linear_dim", None)
        self.attention_dim = int(attn_dim) if attn_dim else head_dim
        self.linear_dim = int(lin_dim) if lin_dim else head_dim
        eps = config.get("epsilon", 1e-6)

        # Reference input preprocessor: learnable absolute position embedding.
        self.pos_embedding = nn.Embedding(config.max_seq_length, self.embedding_dim)
        truncated_normal(
            self.pos_embedding.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self.embedding_dim),
        )
        self.emb_dropout = nn.Dropout(p=self._dropout_rate)

        self.hstu_blocks = nn.ModuleList(
            [
                SequentialTransductionUnit(
                    embedding_dim=self.embedding_dim,
                    linear_dim=self.linear_dim,
                    attention_dim=self.attention_dim,
                    num_heads=config.num_heads,
                    max_seq_len=config.max_seq_length,
                    dropout=self._dropout_rate,
                    concat_ua=self.concat_ua,
                    use_time_bias=self.use_time_bias,
                    num_time_buckets=num_time_buckets,
                    eps=eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        # invalid_attn_mask: 1 = keep (causal lower-triangular incl. diagonal).
        self.register_buffer(
            "invalid_attn_mask",
            (
                1.0
                - torch.triu(
                    torch.ones(
                        config.max_seq_length,
                        config.max_seq_length,
                        dtype=torch.float,
                    ),
                    diagonal=1,
                )
            ),
        )
        self._init_weights()

    def get_hidden_states(self, sequences, sequence_lengths):
        B, N = sequences.size()
        item_embs = self.get_item_embedding(sequences)
        positions = torch.arange(N, device=sequences.device).unsqueeze(0).expand(B, -1)
        x = item_embs * (self.embedding_dim**0.5) + self.pos_embedding(positions)
        x = self.emb_dropout(x)
        valid_mask = (sequences != 0).unsqueeze(-1).float()
        x = x * valid_mask

        timestamps = None
        if self.use_time_bias:
            timestamps = self.get_batch_timestamps()
            if timestamps is not None:
                timestamps = timestamps.to(sequences.device).float()

        mask = self.invalid_attn_mask[:N, :N]
        for block in self.hstu_blocks:
            x = block(x, mask, timestamps=timestamps)
        return x

    def forward(self, sequences, sequence_lengths):
        hidden_states = self.get_hidden_states(sequences, sequence_lengths)
        logits = torch.matmul(
            hidden_states, self.get_output_embeddings().transpose(0, 1)
        )
        return logits

    def predict_next(self, sequences, sequence_lengths):
        logits = self.forward(sequences, sequence_lengths)
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_indices = torch.clamp(
            sequence_lengths - 1, min=0, max=sequences.size(1) - 1
        )
        last_logits = logits[batch_indices, last_indices]
        return torch.softmax(last_logits, dim=-1)

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
        batch_size, seq_len = sequences.size()
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        mask = (positions >= 1) & (positions < sequence_lengths.unsqueeze(1))
        return targets, mask

    def get_loss_mask(self, batch):
        sequences = batch["sequence"]
        sequence_lengths = batch["sequence_length"]
        batch_size, seq_len = sequences.size()
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        mask = ((positions >= 1) & (positions < sequence_lengths.unsqueeze(1))).float()
        return mask

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        # uvqk is already normal(0, 0.02) and W_o xavier in the block __init__.
