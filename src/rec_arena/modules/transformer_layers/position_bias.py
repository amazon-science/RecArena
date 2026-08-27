"""Additive attention-bias position encodings for the SASRec ablation study.

These are the *additive-bias* family of position encodings (as opposed to the
input-additive learnable embedding or the rotary RoPE family). Each module
returns a bias tensor broadcastable to ``[1, num_heads, seq_len, seq_len]`` that
is added to the raw attention scores (pre-softmax). Causal masking is applied
separately in the attention module, so these return the *unmasked* bias.

  - ``ALiBiBias``           : fixed linear distance penalty (no parameters).
  - ``T5RelativeAttentionBias`` : learnable bucketed relative-position bias.
"""

import math

import torch
import torch.nn as nn


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    """ALiBi per-head slopes (geometric sequence), following Press et al. 2022.

    For a power-of-two head count the slopes are 2^{-8k/n} for head k. For
    non-power-of-two counts we use the paper's interpolation trick so the slopes
    still span the same range.
    """

    def _pow2_slopes(n: int) -> list:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start**i for i in range(1, n + 1)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(_pow2_slopes(num_heads), dtype=torch.float32)

    # Nearest lower power of two, then fill the remainder from the 2n sequence.
    closest = 2 ** math.floor(math.log2(num_heads))
    slopes = _pow2_slopes(closest)
    extra = _pow2_slopes(2 * closest)[0::2][: num_heads - closest]
    return torch.tensor(slopes + extra, dtype=torch.float32)


class ALiBiBias(nn.Module):
    """Attention with Linear Biases (ALiBi), Press et al. 2022.

    Adds ``-slope_h * (i - j)`` to the score of query ``i`` attending to key
    ``j`` (parameter-free; slopes fixed per head). Under causal masking only
    ``j <= i`` survive, so the penalty grows with how far back the key is.
    """

    def __init__(self, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        slopes = _alibi_slopes(num_heads)  # [H]
        # Relative distance matrix (i - j), clamped to >=0 for the causal half.
        pos = torch.arange(max_seq_len)
        dist = pos[:, None] - pos[None, :]  # [S, S], >0 for past keys
        bias = -dist.clamp(min=0).float()[None, :, :] * slopes[:, None, None]
        # [1, H, S, S]
        self.register_buffer("bias", bias.unsqueeze(0))

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.bias[:, :, :seq_len, :seq_len]


class T5RelativeAttentionBias(nn.Module):
    """Learnable bucketed relative-position bias (T5 / Raffel et al. 2020).

    Buckets the signed relative distance ``i - j`` (causal: only past keys) into
    ``num_buckets`` log-spaced buckets and looks up a learnable per-head scalar.
    This is the same additive-bias family used by the HSTU/FuXi positional
    channel, so it ties the benchmark's models to a shared PE.
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)
        self.register_buffer(
            "_rel_pos", self._relative_positions(max_seq_len), persistent=False
        )

    def _relative_positions(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len)
        # memory_pos - query_pos: negative for past keys (causal).
        rel = pos[None, :] - pos[:, None]  # [S, S]
        return self._bucketize(rel)

    def _bucketize(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Causal (unidirectional) T5 bucketization.

        All valid keys are in the past (relative_position <= 0). We negate to a
        non-negative distance, send the first half of the buckets to exact small
        distances and the rest to log-spaced larger distances.
        """
        num_buckets = self.num_buckets
        n = (-relative_position).clamp(min=0)  # distance into the past, >=0

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_large = torch.clamp(val_large, max=num_buckets - 1)
        return torch.where(is_small, n, val_large)

    def forward(self, seq_len: int) -> torch.Tensor:
        buckets = self._rel_pos[:seq_len, :seq_len].to(
            self.relative_attention_bias.weight.device
        )
        # [S, S, H] -> [1, H, S, S]
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)
