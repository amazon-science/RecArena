"""Core pack building blocks for batching multiple neural networks into padded tensors."""

from __future__ import annotations

import contextlib
import itertools
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "PACK_DIM",
    "BATCH_DIM",
    "ParameterPack",
    "BufferPack",
    "get_pack_size",
    "make_keep_pack_idx",
    "LinearPack",
    "DropoutPack",
    "LayerNormPack",
    "TransformerBlockPack",
    "module_pack_remove",
    "module_pack_load_state_dict",
    "module_pack_select",
]

# Dimension conventions: PACK_DIM=0, BATCH_DIM=1.
# For linear layers: tensors are (pack_size, batch_size, dim).
# For transformer use: tensors are (pack_size, batch_size, seq_len, dim).
PACK_DIM = 0
BATCH_DIM = 1

# Per-member FFN activations (a diversity lever for the pack). Ordered; the
# index is the id stored per member. gelu is the default at id 0.
_ACT_FNS = [F.gelu, F.relu, F.silu, torch.tanh]
_ACT_NAMES = ["gelu", "relu", "silu", "tanh"]
_ACT_CODE_TO_ID = {n: i for i, n in enumerate(_ACT_NAMES)}


def _apply_pack_activation(h: Tensor, act_ids: Optional[Tensor]) -> Tensor:
    """Apply a per-member activation to h (cur_P, B*S, F).

    act_ids is (cur_P,) ids into _ACT_FNS, or None => all GELU. Computed as a
    masked blend so it stays a single batched op (no python loop over members).
    Only activations actually present in act_ids are evaluated.
    """
    if act_ids is None:
        return F.gelu(h)
    out = None
    for aid in torch.unique(act_ids).tolist():
        mask = (act_ids == aid).view(-1, 1, 1).to(h.dtype)  # (cur_P,1,1)
        contrib = _ACT_FNS[aid](h) * mask
        out = contrib if out is None else out + contrib
    return out


def get_pack_size(x: Tensor) -> int:
    """Returns the size of the pack dimension."""
    return x.shape[PACK_DIM]


def make_keep_pack_idx(pack_size: int, remove_idx: Tensor) -> Tensor:
    """Compute complement of remove indices within [0, pack_size)."""
    device = remove_idx.device
    mask = torch.ones(pack_size, dtype=torch.bool, device=device)
    mask[remove_idx] = False
    return torch.arange(pack_size, device=device)[mask].clone()


# =============================================================================
# Marker classes for pack parameters and buffers
# =============================================================================


class ParameterPack(nn.Parameter):
    """Marker class for pack parameters."""

    pass


class BufferPack(Tensor):
    """Marker for pack buffers. Use with register_buffer inside a ModulePack."""
    pass


def _is_buffer_pack(t: object) -> bool:
    """Check if a tensor is a BufferPack."""
    return isinstance(t, Tensor) and getattr(t, "_is_buffer_pack", False)


class _ModulePackBuffers(OrderedDict):
    """Ensures buffer reassignment preserves BufferPack marker."""

    def __setitem__(self, key, value) -> None:
        current_value = self.get(key)
        if current_value is not None and getattr(current_value, "_is_buffer_pack", False):
            if not getattr(value, "_is_buffer_pack", False):
                value._is_buffer_pack = True
        return super().__setitem__(key, value)


def _get_tensor_pack(x: Optional[Tensor], pack_idx: Optional[Tensor]) -> Optional[Tensor]:
    """Index into pack dimension if pack_idx is provided."""
    if x is None:
        return None
    if pack_idx is None:
        return x
    return x[pack_idx]


# =============================================================================
# LinearPack
# =============================================================================


class LinearPack(nn.Module):
    """Batched linear layer: weight shape (pack_size, max_out, max_in)."""

    def __init__(
        self,
        in_features: int | list[int],
        out_features: int | list[int],
        bias: bool = True,
        *,
        max_in_features: Optional[int] = None,
        max_out_features: Optional[int] = None,
        pack_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # Handle variable in_features
        if isinstance(in_features, list):
            assert len(in_features) == pack_size
            assert max_in_features is not None
            assert all(0 < d <= max_in_features for d in in_features)
            buf = torch.tensor(in_features, dtype=torch.float, device=device)
            buf._is_buffer_pack = True
            self.register_buffer("_in_features", buf)
        else:
            if max_in_features is None:
                max_in_features = in_features
            self._in_features = None

        # Handle variable out_features
        if isinstance(out_features, list):
            assert len(out_features) == pack_size
            assert max_out_features is not None
            assert all(0 < d <= max_out_features for d in out_features)
            buf = torch.tensor(out_features, dtype=torch.float, device=device)
            buf._is_buffer_pack = True
            self.register_buffer("_out_features", buf)
        else:
            if max_out_features is None:
                max_out_features = out_features
            self._out_features = None

        self._max_in_features = max_in_features
        self._max_out_features = max_out_features
        self._pack_size = pack_size

        self.weight = ParameterPack(
            torch.empty(pack_size, max_out_features, max_in_features, **factory_kwargs)
        )
        self.bias = (
            ParameterPack(torch.empty(pack_size, max_out_features, **factory_kwargs))
            if bias
            else None
        )

        self.reset_parameters()

    @property
    def pack_size(self) -> int:
        return self._pack_size

    @property
    def max_in_features(self) -> int:
        return self._max_in_features

    @property
    def max_out_features(self) -> int:
        return self._max_out_features

    def reset_parameters(self) -> None:
        """Xavier uniform for weights, zeros for biases."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, pack_idx: Optional[Tensor] = None) -> Tensor:
        """Forward: (P, B, in) -> (P, B, out) using torch.bmm."""
        assert x.ndim == 3
        assert x.shape[-1] == self._max_in_features

        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)
        out_features = _get_tensor_pack(self._out_features, pack_idx) if self._out_features is not None else None

        # bmm: (P, B, in) @ (P, in, out) -> (P, B, out)
        x = torch.bmm(x, weight.transpose(-2, -1))
        if bias is not None:
            x = x + bias.unsqueeze(BATCH_DIM)

        # Mask output if variable out_features
        if out_features is not None:
            output_mask = (
                torch.arange(self._max_out_features, device=x.device)[None]
                < out_features[:, None]
            )
            x = x * output_mask.float().unsqueeze(BATCH_DIM)

        return x


# =============================================================================
# DropoutPack
# =============================================================================


class DropoutPack(nn.Module):
    """Per-member dropout rates stored as a buffer."""

    def __init__(self, p: float | list[float], *, pack_size: int) -> None:
        super().__init__()
        self._buffers = _ModulePackBuffers()
        self._pack_size = pack_size
        if isinstance(p, float):
            self._scalar_p: Optional[float] = p
            self._p_buffer: Optional[Tensor] = None
        else:
            assert len(p) == pack_size
            assert all(0.0 <= pi <= 1.0 for pi in p)
            self._scalar_p = None
            buf = torch.tensor(p, dtype=torch.float)
            buf._is_buffer_pack = True
            self.register_buffer("_p_buffer", buf)

    @property
    def pack_size(self) -> int:
        return self._pack_size

    def forward(self, x: Tensor, pack_idx: Optional[Tensor] = None) -> Tensor:
        """Per-element Bernoulli with per-member p."""
        if not self.training:
            return x

        if self._scalar_p is not None:
            return F.dropout(x, p=self._scalar_p, training=True)

        p = _get_tensor_pack(self._p_buffer, pack_idx)
        assert p is not None
        p_keep = 1.0 - p
        view_shape = (p_keep.shape[0],) + (1,) * (x.ndim - 1)
        p_keep = p_keep.view(*view_shape).expand_as(x)
        mask = torch.bernoulli(p_keep)
        return x * mask / p_keep.clamp(min=1e-8)


# =============================================================================
# LayerNormPack
# =============================================================================


class LayerNormPack(nn.Module):
    """Per-member layer norm. Weight/bias shape (pack_size, dim). Normalizes last dim."""

    def __init__(self, dim: int, *, pack_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self._dim = dim
        self._pack_size = pack_size
        self._eps = eps

        self.weight = ParameterPack(torch.ones(pack_size, dim))
        self.bias = ParameterPack(torch.zeros(pack_size, dim))

    @property
    def pack_size(self) -> int:
        return get_pack_size(self.weight)

    def reset_parameters(self) -> None:
        """Ones for norm weights, zeros for biases."""
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, pack_idx: Optional[Tensor] = None) -> Tensor:
        """Normalize the last dimension per-member."""
        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)

        # Normalize the last dim with the fused kernel (no affine here), then
        # apply the per-member affine. F.layer_norm is much faster on MPS/CUDA
        # than a hand-rolled mean/var/rsqrt.
        x_norm = F.layer_norm(x, (x.shape[-1],), eps=self._eps)

        # Broadcast weight and bias: (P, D) -> (P, 1, ..., 1, D)
        n_middle_dims = x.ndim - 2
        view_shape = (weight.shape[0],) + (1,) * n_middle_dims + (weight.shape[-1],)
        return torch.addcmul(
            bias.view(*view_shape), x_norm, weight.view(*view_shape)
        )


# =============================================================================
# TransformerBlockPack
# =============================================================================


class TransformerBlockPack(nn.Module):
    """One transformer block with per-member feedforward_dim and dropout rates."""

    def __init__(
        self,
        dim: int,
        feedforward_dim: int | list[int],
        dropout: float | list[float] = 0.0,
        *,
        num_heads: int = 1,
        max_feedforward_dim: Optional[int] = None,
        activations: Optional[list[str]] = None,
        pack_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self._buffers = _ModulePackBuffers()
        factory_kwargs = {"device": device, "dtype": dtype}

        assert dim % num_heads == 0
        self._dim = dim
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._pack_size = pack_size

        # Per-member FFN activation (diversity lever). Codes -> id in _ACT_FNS.
        # None => all GELU. Applied via masked blend in _ffn so it stays batched.
        self._act_codes = activations
        if activations is not None:
            assert len(activations) == pack_size
            act_ids = torch.tensor(
                [_ACT_CODE_TO_ID[a] for a in activations], dtype=torch.long, device=device
            )
            act_ids._is_buffer_pack = True
            self.register_buffer("_act_ids", act_ids)
        else:
            self._act_ids = None

        # Handle variable feedforward_dim
        if isinstance(feedforward_dim, list):
            assert len(feedforward_dim) == pack_size
            assert max_feedforward_dim is not None
            assert all(0 < d <= max_feedforward_dim for d in feedforward_dim)
            buf = torch.tensor(feedforward_dim, dtype=torch.float, device=device)
            buf._is_buffer_pack = True
            self.register_buffer("_ffn_dim", buf)
        else:
            if max_feedforward_dim is None:
                max_feedforward_dim = feedforward_dim
            self._ffn_dim = None

        self._max_feedforward_dim = max_feedforward_dim

        # Pre-norm layers
        self.norm1 = LayerNormPack(dim, pack_size=pack_size)
        self.norm2 = LayerNormPack(dim, pack_size=pack_size)

        # Attention: QKV weight (P, 3*dim, dim), out weight (P, dim, dim)
        self.qkv_weight = ParameterPack(
            torch.empty(pack_size, 3 * dim, dim, **factory_kwargs)
        )
        self.qkv_bias = ParameterPack(
            torch.empty(pack_size, 3 * dim, **factory_kwargs)
        )
        self.out_weight = ParameterPack(
            torch.empty(pack_size, dim, dim, **factory_kwargs)
        )
        self.out_bias = ParameterPack(
            torch.empty(pack_size, dim, **factory_kwargs)
        )

        # FFN weights: (P, max_ffn, dim) and (P, dim, max_ffn)
        self.ffn_up_weight = ParameterPack(
            torch.empty(pack_size, max_feedforward_dim, dim, **factory_kwargs)
        )
        self.ffn_up_bias = ParameterPack(
            torch.empty(pack_size, max_feedforward_dim, **factory_kwargs)
        )
        self.ffn_down_weight = ParameterPack(
            torch.empty(pack_size, dim, max_feedforward_dim, **factory_kwargs)
        )
        self.ffn_down_bias = ParameterPack(
            torch.empty(pack_size, dim, **factory_kwargs)
        )

        # Dropout
        self.attn_dropout = DropoutPack(dropout, pack_size=pack_size)
        self.ffn_dropout = DropoutPack(dropout, pack_size=pack_size)

        self.reset_parameters()

    @property
    def pack_size(self) -> int:
        return get_pack_size(self.qkv_weight)

    def reset_parameters(self) -> None:
        """Xavier uniform for weight matrices, zeros for biases."""
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.zeros_(self.qkv_bias)
        nn.init.xavier_uniform_(self.out_weight)
        nn.init.zeros_(self.out_bias)
        nn.init.xavier_uniform_(self.ffn_up_weight)
        nn.init.zeros_(self.ffn_up_bias)
        nn.init.xavier_uniform_(self.ffn_down_weight)
        nn.init.zeros_(self.ffn_down_bias)

    def _attention(
        self,
        x: Tensor,
        pack_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head self-attention using SDPA. x: (P, B, S, D)."""
        P, B, S, D = x.shape
        H = self._num_heads
        HD = self._head_dim

        qkv_w = _get_tensor_pack(self.qkv_weight, pack_idx)
        qkv_b = _get_tensor_pack(self.qkv_bias, pack_idx)
        out_w = _get_tensor_pack(self.out_weight, pack_idx)
        out_b = _get_tensor_pack(self.out_bias, pack_idx)

        cur_P = qkv_w.shape[0]

        # QKV projection: (P, B*S, D) @ (P, D, 3D) -> (P, B*S, 3D)
        x_flat = x.reshape(cur_P, B * S, D)
        qkv = torch.bmm(x_flat, qkv_w.transpose(-2, -1))
        qkv = qkv + qkv_b.unsqueeze(1)

        # Reshape for multi-head: (P, B, S, 3, H, HD) -> (3, P, B, H, S, HD)
        qkv = qkv.reshape(cur_P, B, S, 3, H, HD).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Merge pack into heads for SDPA: (B, P*H, S, HD)
        # q/k/v are (P, B, H, S, HD). SDPA is independent per head, so
        # folding P into the head dim is equivalent but faster on MPS.
        q = q.permute(1, 0, 2, 3, 4).reshape(B, cur_P * H, S, HD)
        k = k.permute(1, 0, 2, 3, 4).reshape(B, cur_P * H, S, HD)
        v = v.permute(1, 0, 2, 3, 4).reshape(B, cur_P * H, S, HD)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape back: (B, P*H, S, HD) -> (P, B*S, D)
        attn_out = attn_out.reshape(B, cur_P, H, S, HD)
        attn_out = attn_out.permute(1, 0, 3, 2, 4).reshape(cur_P, B * S, D)

        # Output projection: (P, B*S, D) @ (P, D, D) -> (P, B*S, D)
        out = torch.bmm(attn_out, out_w.transpose(-2, -1))
        out = out + out_b.unsqueeze(1)

        return out.reshape(cur_P, B, S, D)

    def _ffn(
        self,
        x: Tensor,
        pack_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Feed-forward network. x: (P, B, S, D)."""
        P, B, S, D = x.shape

        ffn_up_w = _get_tensor_pack(self.ffn_up_weight, pack_idx)
        ffn_up_b = _get_tensor_pack(self.ffn_up_bias, pack_idx)
        ffn_down_w = _get_tensor_pack(self.ffn_down_weight, pack_idx)
        ffn_down_b = _get_tensor_pack(self.ffn_down_bias, pack_idx)
        ffn_dim = _get_tensor_pack(self._ffn_dim, pack_idx) if self._ffn_dim is not None else None

        cur_P = ffn_up_w.shape[0]
        max_ffn = self._max_feedforward_dim

        # Up projection: (P, B*S, D) @ (P, D, max_ffn) -> (P, B*S, max_ffn)
        x_flat = x.reshape(cur_P, B * S, D)
        h = torch.bmm(x_flat, ffn_up_w.transpose(-2, -1))
        h = h + ffn_up_b.unsqueeze(1)

        # Per-member activation (diversity lever); GELU if not configured.
        act_ids = _get_tensor_pack(self._act_ids, pack_idx) if self._act_ids is not None else None
        h = _apply_pack_activation(h, act_ids)

        # Mask if variable ffn_dim
        if ffn_dim is not None:
            ffn_mask = (
                torch.arange(max_ffn, device=h.device)[None]
                < ffn_dim[:, None]
            ).float()  # (cur_P, max_ffn)
            h = h * ffn_mask.unsqueeze(1)

        # Dropout (reshape to 4D for DropoutPack)
        h_pack = h.reshape(cur_P, B, S, max_ffn)
        h_pack = self.ffn_dropout(h_pack, pack_idx)
        h = h_pack.reshape(cur_P, B * S, max_ffn)

        # Down projection: (P, B*S, max_ffn) @ (P, max_ffn, D) -> (P, B*S, D)
        out = torch.bmm(h, ffn_down_w.transpose(-2, -1))
        out = out + ffn_down_b.unsqueeze(1)

        return out.reshape(cur_P, B, S, D)

    def forward(self, x: Tensor, pack_idx: Optional[Tensor] = None) -> Tensor:
        """Forward: x is (P, B, S, D) -> (P, B, S, D). Pre-norm architecture."""
        # Self-attention with pre-norm
        residual = x
        x_normed = self.norm1(x, pack_idx)
        attn_out = self._attention(x_normed, pack_idx)
        attn_out = self.attn_dropout(attn_out, pack_idx)
        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x_normed = self.norm2(x, pack_idx)
        ffn_out = self._ffn(x_normed, pack_idx)
        x = residual + ffn_out

        return x

    def forward_subset(self, x: Tensor, pack_idx: Tensor) -> Tensor:
        """Forward for a subset of members (variable depth support)."""
        return self.forward(x, pack_idx=pack_idx)


# =============================================================================
# Pack utilities
# =============================================================================


def module_pack_remove(module: nn.Module, pack_idx: Tensor) -> dict[ParameterPack, ParameterPack]:
    """Remove members from all ParameterPack/BufferPack tensors. Returns old_to_new mapping for optimizer fixup."""
    assert len(pack_idx) > 0

    keep_pack_idx: Optional[Tensor] = None
    old_to_new: dict[ParameterPack, ParameterPack] = {}

    # Collect all items first to avoid mutating during iteration
    items = []
    for name, x in itertools.chain(
        list(module.named_parameters()), list(module.named_buffers())
    ):
        is_parameter_pack = isinstance(x, ParameterPack)
        is_buffer_pack = _is_buffer_pack(x)
        if is_parameter_pack or is_buffer_pack:
            items.append((name, x, is_parameter_pack))

    for name, x, is_parameter_pack in items:
        if keep_pack_idx is None:
            keep_pack_idx = make_keep_pack_idx(get_pack_size(x), pack_idx)

        new_data = x.data[keep_pack_idx].clone()
        if "." in name:
            submodule_name, attr = name.rsplit(".", 1)
            submodule = module.get_submodule(submodule_name)
        else:
            submodule = module
            attr = name

        if is_parameter_pack:
            new_x = ParameterPack(new_data)
            setattr(submodule, attr, new_x)
            old_to_new[x] = new_x  # type: ignore[assignment]
        else:
            new_data._is_buffer_pack = True
            if hasattr(submodule, attr):
                delattr(submodule, attr)
            submodule.register_buffer(attr, new_data)

    return old_to_new


def module_pack_load_state_dict(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    pack_idx: Tensor,
    *,
    state_dict_idx: Optional[Tensor] = None,
) -> None:
    """Load state dict into specific members of a module pack."""
    state_dict = state_dict.copy()
    for name, x in itertools.chain(module.named_parameters(), module.named_buffers()):
        if isinstance(x, ParameterPack) or _is_buffer_pack(x):
            if name in state_dict:
                src_idx = pack_idx if state_dict_idx is None else state_dict_idx
                x.data[pack_idx] = state_dict.pop(name)[src_idx]


@contextmanager
def module_pack_select(module: nn.Module, pack_idx: Tensor) -> Iterator[None]:
    """Context manager that temporarily selects a subset of pack members."""
    original_state: list[tuple[nn.Module, str, Tensor]] = []

    if pack_idx is not None:
        for name, x in itertools.chain(
            module.named_parameters(), module.named_buffers()
        ):
            is_parameter_pack = isinstance(x, ParameterPack)
            is_buffer_pack = _is_buffer_pack(x)
            if is_parameter_pack or is_buffer_pack:
                if "." in name:
                    submodule_name, attr = name.rsplit(".", 1)
                    submodule = module.get_submodule(submodule_name)
                else:
                    submodule = module
                    attr = name

                if is_parameter_pack:
                    new_x = ParameterPack(x.data[pack_idx])
                else:
                    new_x = BufferPack(x.data[pack_idx])

                setattr(submodule, attr, new_x)
                original_state.append((submodule, attr, x))

    try:
        yield
    finally:
        for submodule, attr, original_tensor in original_state:
            setattr(submodule, attr, original_tensor)
