"""Custom optimizer packs for the TabPack pattern applied to sequential recommendation.

Inspired by TabPack (Gorishniy et al., ICML 2026). Each "pack member" (a separate model
variant) can have its own learning rate, weight decay, beta1, beta2, etc. -- stored as
per-member tensors along dimension 0.
"""

import functools
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT


__all__ = [
    "AdamWPack",
    "MuonPack",
    "MuonAdamWPack",
    "zeropower_via_newtonschulz5",
    "optimizer_pack_remove",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_valid_lr(value: float) -> bool:
    return value >= 0.0


def _is_valid_beta(value: float) -> bool:
    return 0.0 <= value < 1.0


def _is_valid_weight_decay(value: float) -> bool:
    return value >= 0.0


def _is_valid_eps(value: float) -> bool:
    return value > 0.0


def _is_shared_group_value(value) -> bool:
    """Check if the group value is shared between pack members."""
    return value is None or isinstance(
        value, bool | int | float | str | bytes | tuple | dict
    )


# ---------------------------------------------------------------------------
# Broadcasting helper
# ---------------------------------------------------------------------------


def _maybe_unsqueeze(value, *, p):
    """Unsqueeze a per-member tensor to broadcast against parameter p.

    If value is a Tensor of shape (pack_size,), unsqueeze to
    (pack_size, 1, 1, ...) matching p.ndim. If scalar, return as-is.
    """
    return value[:, *((None,) * (p.ndim - 1))] if isinstance(value, Tensor) else value


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization (from the Muon paper)
# ---------------------------------------------------------------------------


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the slope
    at zero. Supports batched inputs of shape (pack_size, m, n).

    Reference: https://github.com/KellerJordan/Muon
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# ---------------------------------------------------------------------------
# Weight decay multiplier helper
# ---------------------------------------------------------------------------


def _make_weight_decay_multiplier(
    *, lr: float | Tensor, weight_decay: float | Tensor
) -> None | float | Tensor:
    if (isinstance(lr, float) and lr == 0.0) or (
        isinstance(weight_decay, float) and weight_decay == 0.0
    ):
        return None
    return 1 - lr * weight_decay


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _OptimizerPackBase(torch.optim.Optimizer):
    """Base class for optimizer packs.

    Analogously to a module pack, an optimizer pack represents a set of optimizers
    with potentially different hyperparameters (learning rate, weight decay, etc.).
    """

    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None:
        super().__init__(
            params,
            {
                # torch.tensor is used to ensure that each group receives a tensor
                # with a separate storage.
                k: v if _is_shared_group_value(v) else torch.tensor(v)
                for k, v in defaults.items()
            },
        )

        for group in self.param_groups:
            group_params = group["params"]
            if not group_params:
                continue
            device = group_params[0].device
            for key, value in list(group.items()):
                if key != "params" and isinstance(value, Tensor):
                    group[key] = value.to(device=device)


# ---------------------------------------------------------------------------
# AdamWPack
# ---------------------------------------------------------------------------


class AdamWPack(_OptimizerPackBase):
    """A pack of AdamW optimizers.

    Each hyperparameter can be a float (shared across all pack members) or a
    list[float] of length pack_size (per-member). Lists are converted to tensors
    and stored in param_groups.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float | list[float],
        beta1: float | list[float] = 0.9,
        beta2: float | list[float] = 0.999,
        eps: float | list[float] = 1e-8,
        weight_decay: float | list[float],
        pack_size: int,
    ):
        assert pack_size > 0

        defaults: dict[str, Any] = {}
        for key, value, is_valid_fn in [
            ("lr", lr, _is_valid_lr),
            ("beta1", beta1, _is_valid_beta),
            ("beta2", beta2, _is_valid_beta),
            ("eps", eps, _is_valid_eps),
            ("weight_decay", weight_decay, _is_valid_weight_decay),
        ]:
            if isinstance(value, list):
                assert len(value) == pack_size
                assert all(map(is_valid_fn, value))
            else:
                assert is_valid_fn(value)
            defaults[key] = value

        defaults["pack_size"] = pack_size
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: None | Callable[[], Tensor] = None,
    ) -> None | Tensor:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            pack_size = group["pack_size"]

            weight_decay_multiplier = _make_weight_decay_multiplier(
                lr=lr, weight_decay=group["weight_decay"]
            )

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                assert not grad.is_sparse, "Sparse gradients are not supported"

                maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros(
                        pack_size, dtype=torch.int64, device=p.device
                    )
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1

                # Decoupled weight decay
                if weight_decay_multiplier is not None:
                    p.mul_(maybe_unsqueeze(weight_decay_multiplier))

                # Update biased first moment estimate
                exp_avg.lerp_(grad, maybe_unsqueeze(1 - beta1))

                # Update biased second raw moment estimate
                if isinstance(beta2, float):
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                else:
                    exp_avg_sq.lerp_(grad.square(), maybe_unsqueeze(1 - beta2))

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = lr / bias_correction1

                if isinstance(bias_correction2, float):
                    denom = exp_avg_sq.sqrt() / bias_correction2**0.5
                else:
                    denom = exp_avg_sq.sqrt().div_(
                        maybe_unsqueeze(bias_correction2**0.5)
                    )
                denom.add_(maybe_unsqueeze(eps))

                if isinstance(step_size, float):
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p.sub_((exp_avg / denom).mul_(maybe_unsqueeze(step_size)))

        return loss


# ---------------------------------------------------------------------------
# MuonPack
# ---------------------------------------------------------------------------


class MuonPack(_OptimizerPackBase):
    """A pack of Muon optimizers (for backbone linear weights only).

    Muon -- MomentUm Orthogonalized by Newton-schulz. Each pack member can have
    its own learning rate, momentum, and weight decay.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float | list[float],
        momentum: float | list[float] = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float | list[float] = 0.0,
        pack_size: int,
    ):
        assert pack_size > 0

        defaults: dict[str, Any] = {
            "ns_steps": ns_steps,
            "nesterov": nesterov,
            "muon_scale": None,
        }
        for key, value, is_valid_fn in [
            ("lr", lr, _is_valid_lr),
            ("momentum", momentum, _is_valid_beta),
            ("weight_decay", weight_decay, _is_valid_weight_decay),
        ]:
            if isinstance(value, list):
                assert len(value) == pack_size
                assert all(map(is_valid_fn, value))
            else:
                assert is_valid_fn(value)
            defaults[key] = value

        defaults["pack_size"] = pack_size
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: None | Callable[[], Tensor] = None,
    ) -> None | Tensor:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            muon_scale = group["muon_scale"]

            weight_decay_multiplier = _make_weight_decay_multiplier(
                lr=lr, weight_decay=weight_decay
            )

            for p in group["params"]:
                # Muon expects 2D weights; with pack dim: (pack_size, m, n)
                assert p.ndim == 3
                if p.grad is None:
                    continue

                grad = p.grad.data
                assert not grad.is_sparse, "Sparse gradients are not supported"

                maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                momentum_buffer: Tensor = state["momentum_buffer"]

                # Decoupled weight decay
                if weight_decay_multiplier is not None:
                    p.mul_(maybe_unsqueeze(weight_decay_multiplier))

                # Momentum buffer update
                momentum_buffer.lerp_(grad, maybe_unsqueeze(1 - momentum))

                # Nesterov update
                if nesterov:
                    update = grad.lerp_(momentum_buffer, maybe_unsqueeze(momentum))
                else:
                    update = momentum_buffer

                # Newton-Schulz orthogonalization
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Scale by max(1, m/n)**0.5
                if muon_scale is None:
                    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
                else:
                    update.mul_(muon_scale.view(-1, 1, 1))

                # Parameter update
                if isinstance(lr, float):
                    p.sub_(update, alpha=lr)
                else:
                    p.sub_(update.mul_(maybe_unsqueeze(lr)))

        return loss


# ---------------------------------------------------------------------------
# MuonAdamWPack
# ---------------------------------------------------------------------------


class MuonAdamWPack(_OptimizerPackBase):
    """A pack combining Muon (for backbone weights) and AdamW (for everything else).

    Each param_group has a 'muon': bool flag. Groups with muon=True use Muon step,
    others use AdamW step. Also supports muon_scale per group (a tensor of
    (pack_size,) for per-member m/n ratio correction).
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float | list[float],
        beta1: float | list[float] = 0.9,
        beta2: float | list[float] = 0.999,
        eps: float | list[float] = 1e-8,
        weight_decay: float | list[float],
        muon_lr: float | list[float],
        muon_momentum: float | list[float] = 0.95,
        muon_ns_steps: int = 5,
        muon_nesterov: bool = True,
        pack_size: int,
    ):
        assert pack_size > 0

        defaults: dict[str, Any] = {
            "muon": False,
            "muon_ns_steps": muon_ns_steps,
            "muon_nesterov": muon_nesterov,
            # Per-pack-member spectral correction max(1, m_i/n_i)**0.5.
            # Override per param group with a (pack_size,) tensor when the
            # parameter pack zero-pads matrices of different actual shapes;
            # falls back to the padded global shape when None.
            "muon_scale": None,
        }
        for key, value, is_valid_fn in [
            ("lr", lr, _is_valid_lr),
            ("beta1", beta1, _is_valid_beta),
            ("beta2", beta2, _is_valid_beta),
            ("eps", eps, _is_valid_eps),
            ("weight_decay", weight_decay, _is_valid_weight_decay),
            ("muon_lr", muon_lr, _is_valid_lr),
            ("muon_momentum", muon_momentum, _is_valid_beta),
        ]:
            if isinstance(value, list):
                assert len(value) == pack_size
                assert all(map(is_valid_fn, value))
            else:
                assert is_valid_fn(value)
            defaults[key] = value

        defaults["pack_size"] = pack_size
        super().__init__(params, defaults)

    def _step_muon(self, group: dict[str, Any]) -> None:
        lr = group["muon_lr"]
        momentum = group["muon_momentum"]
        weight_decay = group["weight_decay"]
        ns_steps = group["muon_ns_steps"]
        nesterov = group["muon_nesterov"]
        muon_scale = group["muon_scale"]

        if lr is None:
            lr = group["lr"]

        weight_decay_multiplier = _make_weight_decay_multiplier(
            lr=lr, weight_decay=weight_decay
        )

        for p in group["params"]:
            # 3 = 2 layer dimensions + 1 pack dimension
            assert p.ndim == 3
            if p.grad is None:
                continue

            grad = p.grad.data
            assert not grad.is_sparse, "Sparse gradients are not supported"

            maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            momentum_buffer: Tensor = state["momentum_buffer"]

            # Decoupled weight decay
            if weight_decay_multiplier is not None:
                p.mul_(maybe_unsqueeze(weight_decay_multiplier))

            # Momentum buffer update
            momentum_buffer.lerp_(grad, maybe_unsqueeze(1 - momentum))

            # Nesterov update
            if nesterov:
                update = grad.lerp_(momentum_buffer, maybe_unsqueeze(momentum))
            else:
                update = momentum_buffer

            # Newton-Schulz orthogonalization
            update = zeropower_via_newtonschulz5(update, steps=ns_steps)

            # Scale by max(1, m/n)**0.5
            if muon_scale is None:
                update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
            else:
                update.mul_(muon_scale.view(-1, 1, 1))

            assert update.shape == p.shape
            # Parameter update
            if isinstance(lr, float):
                p.sub_(update, alpha=lr)
            else:
                p.sub_(update.mul_(maybe_unsqueeze(lr)))

    def _step_adamw(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        pack_size = group["pack_size"]

        weight_decay_multiplier = _make_weight_decay_multiplier(
            lr=lr, weight_decay=weight_decay
        )

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.data
            assert not grad.is_sparse, "Sparse gradients are not supported"

            maybe_unsqueeze = functools.partial(_maybe_unsqueeze, p=p)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    pack_size, dtype=torch.int64, device=p.device
                )
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            state["step"] += 1

            # Decoupled weight decay
            if weight_decay_multiplier is not None:
                p.mul_(maybe_unsqueeze(weight_decay_multiplier))

            # Update biased first moment estimate
            exp_avg.lerp_(grad, maybe_unsqueeze(1 - beta1))

            # Update biased second raw moment estimate
            if isinstance(beta2, float):
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            else:
                exp_avg_sq.lerp_(grad.square(), maybe_unsqueeze(1 - beta2))

            # Bias correction
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            step_size = lr / bias_correction1

            if isinstance(bias_correction2, float):
                denom = exp_avg_sq.sqrt() / bias_correction2**0.5
            else:
                denom = exp_avg_sq.sqrt().div_(
                    maybe_unsqueeze(bias_correction2**0.5)
                )
            denom.add_(maybe_unsqueeze(eps))

            if isinstance(step_size, float):
                p.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                p.sub_((exp_avg / denom).mul_(maybe_unsqueeze(step_size)))

    @torch.no_grad()
    def step(
        self,
        closure: None | Callable[[], Tensor] = None,
    ) -> None | Tensor:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)

        return loss


# ---------------------------------------------------------------------------
# Optimizer pack utilities
# ---------------------------------------------------------------------------


def _make_keep_pack_idx(pack_size: int, remove_pack_idx: Tensor) -> Tensor:
    """Compute pack indices to keep from pack indices to remove."""
    device = remove_pack_idx.device
    mask = torch.ones(pack_size, dtype=torch.bool, device=device)
    mask[remove_pack_idx] = False
    return torch.arange(pack_size, device=device)[mask].clone()


def optimizer_pack_remove(
    optimizer: torch.optim.Optimizer,
    pack_idx: Tensor,
    old_to_new: dict,
) -> None:
    """Remove pack members from optimizer state.

    Args:
        optimizer: The optimizer pack to modify.
        pack_idx: Indices of pack members to remove.
        old_to_new: Mapping from old parameter packs to new (sliced) parameter packs.
    """
    assert len(pack_idx) > 0
    assert old_to_new

    pack_size = len(next(iter(old_to_new.keys())))
    keep_pack_idx = _make_keep_pack_idx(pack_size, pack_idx)

    new_pack_size = len(keep_pack_idx)
    for group in optimizer.param_groups:
        # Slice per-member hyperparameters (tensors with ndim > 0)
        for key, value in list(group.items()):
            if isinstance(value, Tensor) and value.ndim > 0:
                group[key] = value[keep_pack_idx].clone()
        # Keep the per-group pack_size in sync so any state initialized after a
        # removal (e.g. a block first used post-removal) allocates the right size.
        if "pack_size" in group:
            group["pack_size"] = new_pack_size

        # Swap parameter references and slice state tensors
        for i, p in list(enumerate(group["params"])):
            if p in old_to_new:
                state = optimizer.state.pop(p, None)
                # State can be missing even if optimizer has already been used.
                # For example, some blocks remain unused (and thus don't have
                # the corresponding optimizer states) if the maximum allowed
                # number of blocks is never used.
                if state is not None:
                    for key, value in list(state.items()):
                        if isinstance(value, Tensor) and value.ndim > 0:
                            state[key] = value[keep_pack_idx].clone()
                p_new = old_to_new[p]
                group["params"][i] = p_new
                if state is not None:
                    optimizer.state[p_new] = state
