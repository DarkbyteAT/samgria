"""Reptile meta-optimizer.

Reptile's outer update is parameter interpolation rather than
gradient-based: ``theta += meta_lr * mean(theta'_i - theta)``.  This
avoids the need for query sets entirely — the outer signal comes from
the direction each task's inner loop moved the parameters.

Like FOMAML, the inner loop uses ``create_graph=False`` (no second-order
gradients needed).  A custom inner optimizer can be supplied via
``inner_optimizer_fn``; state isolation is guaranteed by
``save_state`` / ``restore_state``.

References
----------
Nichols, Schulman (2018).  On First-Order Meta-Learning Algorithms.
arXiv:1803.02999.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import vector_to_parameters

from samgria.meta.protocol import (
    InnerOptimizerFn,
    InnerRegFn,
    apply_inner_reg,
    capture_base_params,
)
from samgria.state import AdaptedState, ParameterSnapshot, restore_state, save_state
from samgria.transforms.protocol import GradientTransform


__all__ = ["Reptile"]


class Reptile:
    """Reptile meta-optimizer with parameter interpolation outer update.

    Parameters
    ----------
    inner_lr
        Learning rate for inner-loop SGD steps (used when no custom
        ``inner_optimizer_fn`` is provided).
    meta_lr
        Interpolation step size for the outer update.  Controls how far
        outer parameters move toward the mean of adapted parameters.
    """

    create_graph: bool = False

    def __init__(self, inner_lr: float, meta_lr: float) -> None:
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr

    def adapt(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable[..., T.Tensor],
        support: tuple[T.Tensor, ...],
        inner_steps: int,
        grad_transforms: Sequence[GradientTransform] = (),
        inner_optimizer_fn: InnerOptimizerFn | None = None,
        inner_reg_fn: InnerRegFn | None = None,
    ) -> AdaptedState:
        """Run k inner optimisation steps on support data, return adapted state.

        Identical to FOMAML's inner loop (create_graph=False).  Saves and
        restores the caller's state to maintain full isolation.
        """
        outer_snapshot = save_state(model, optimizer)
        base_params = capture_base_params(model, inner_reg_fn)

        inner_opt = (
            inner_optimizer_fn(model.parameters())
            if inner_optimizer_fn is not None
            else None
        )

        for _ in range(inner_steps):
            loss = apply_inner_reg(
                loss_fn(*support), model, inner_reg_fn, base_params,
            )
            loss.backward()  # pyright: ignore[reportUnknownMemberType]

            for transform in grad_transforms:
                transform.apply(model, loss_fn, support)

            if inner_opt is not None:
                inner_opt.step()
                inner_opt.zero_grad()
            else:
                with T.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.data -= self.inner_lr * p.grad
                model.zero_grad()

        snapshot = save_state(model, optimizer)
        restore_state(model, optimizer, outer_snapshot)
        return AdaptedState(snapshot=snapshot)

    def meta_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        base_snapshot: ParameterSnapshot,
        adapted: Sequence[AdaptedState],
        query_losses: Sequence[T.Tensor] | None = None,
    ) -> None:
        """Apply the Reptile outer update via parameter interpolation.

        Computes ``theta += meta_lr * mean(theta'_i - theta)`` and writes
        the result directly into the model's parameters.  ``query_losses``
        is accepted for protocol compatibility but ignored.
        """
        outer_params = base_snapshot.params
        mean_diff = T.stack(
            [a.snapshot.params - outer_params for a in adapted]
        ).mean(dim=0)

        new_params = outer_params + self.meta_lr * mean_diff
        vector_to_parameters(new_params, model.parameters())
