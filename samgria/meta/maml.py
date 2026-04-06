"""MAML and FOMAML meta-optimizers.

MAML (Model-Agnostic Meta-Learning) runs k inner gradient-descent steps
per task, preserving the full computation graph through each step so that
second-order derivatives flow into the outer update.  The inner loop uses
``torch.func.functional_call`` to evaluate the model with updated parameter
tensors without in-place mutation — this keeps the autograd graph intact
across all inner steps.

FOMAML is the first-order approximation: same inner-loop structure but
the computation graph is not retained, dropping second-order terms for a
cheaper update that is often comparably effective.  Since no graph needs
to flow through, FOMAML uses direct ``.data`` mutation for inner steps.

Both implementations use vanilla SGD for the inner loop by default.  A
custom inner optimizer can be supplied via ``inner_optimizer_fn`` (FOMAML
only); full state isolation is guaranteed by ``save_state`` /
``restore_state`` regardless of which optimizer is used.

References
----------
Finn, Abbeel, Levine (2017).  Model-Agnostic Meta-Learning for Fast
Adaptation of Deep Networks.  ICML.

Nichols, Schulman (2018).  On First-Order Meta-Learning Algorithms.
arXiv:1803.02999.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch as T
import torch.nn as nn
import torch.optim as optim

from samgria.meta.protocol import (
    InnerOptimizerFn,
    InnerRegFn,
    apply_inner_reg,
    capture_base_params,
)
from samgria.state import AdaptedState, ParameterSnapshot, restore_state, save_state
from samgria.transforms.protocol import GradientTransform
from samgria.utils.functional import functional_forward


__all__ = ["FOMAML", "MAML"]


class MAML:
    """Model-Agnostic Meta-Learning with full second-order gradients.

    The inner loop uses ``functional_call`` + ``backward(create_graph=True)``
    to build a differentiable computation graph through every inner SGD
    step.  The ``AdaptedState`` returned by ``adapt()`` carries graph-
    connected ``live_params`` for computing differentiable query losses
    via ``query_forward()``.

    Parameters
    ----------
    inner_lr
        Learning rate for the inner-loop SGD steps.
    """

    create_graph: bool = True

    def __init__(self, inner_lr: float) -> None:
        self.inner_lr = inner_lr

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
        """Run k differentiable inner steps, return adapted state.

        The inner loop builds new parameter tensors at each step via
        ``p_new = p - lr * grad`` (no in-place mutation), keeping the
        full computation graph alive for second-order differentiation.

        The returned ``AdaptedState`` contains both a detached snapshot
        and the graph-connected ``live_params`` dict for computing
        differentiable query losses via ``query_forward()``.
        """
        if inner_optimizer_fn is not None:
            raise ValueError(
                "MAML does not support inner_optimizer_fn — the functional "
                "inner loop requires differentiable SGD steps."
            )

        outer_snapshot = save_state(model, optimizer)
        base_params = capture_base_params(model, inner_reg_fn)

        # Start from the model's current parameters
        params: dict[str, T.Tensor] = {
            k: v.clone().requires_grad_(True)
            for k, v in model.named_parameters()
        }

        for _ in range(inner_steps):
            with functional_forward(model, params):
                loss = loss_fn(*support)

            if inner_reg_fn is not None:
                assert base_params is not None
                loss = loss + inner_reg_fn(params, base_params)

            # Use autograd.grad — works on non-leaf tensors (params after
            # step 1 are non-leaf since they were created by subtraction).
            param_values = list(params.values())
            grads = T.autograd.grad(loss, param_values, create_graph=True)
            grad_dict = dict(zip(params.keys(), grads, strict=True))

            # Bridge grads to model.parameters() for GradientTransform compat
            if grad_transforms:
                for name, p in model.named_parameters():
                    p.grad = grad_dict[name].detach().clone()
                with functional_forward(model, params):
                    for transform in grad_transforms:
                        transform.apply(model, loss_fn, support)
                # Read back any modified gradients from transforms
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        grad_dict[name] = p.grad

            # Differentiable SGD step — creates new tensor nodes
            params = {
                k: p - self.inner_lr * grad_dict[k]
                for k, p in params.items()
            }

        # Build detached snapshot
        with T.no_grad():
            for name, p in model.named_parameters():
                p.copy_(params[name].detach())
        snapshot = save_state(model, optimizer)

        restore_state(model, optimizer, outer_snapshot)
        return AdaptedState(snapshot=snapshot, live_params=params)

    def meta_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        base_snapshot: ParameterSnapshot,
        adapted: Sequence[AdaptedState],
        query_losses: Sequence[T.Tensor] | None = None,
    ) -> None:
        """Compute and apply the second-order outer-loop update.

        Parameters
        ----------
        query_losses
            Scalar losses computed via ``query_forward()`` with
            ``AdaptedState.live_params``.  These carry the computation
            graph through the inner-loop steps, enabling true
            second-order meta-gradients.
        """
        if query_losses is None:
            raise ValueError(
                "MAML.meta_step() requires query_losses computed via "
                "query_forward() (one per adapted state)."
            )
        if len(query_losses) != len(adapted):
            raise ValueError(
                f"Expected {len(adapted)} query losses, "
                f"got {len(query_losses)}."
            )

        optimizer.zero_grad()
        total_loss = T.stack(list(query_losses)).mean()
        total_loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()


class FOMAML:
    """First-Order MAML — drops second-order gradients for efficiency.

    Uses direct ``.data`` mutation for inner-loop steps since no
    computation graph needs to flow through.  Significantly cheaper
    than full MAML while often achieving comparable performance.

    Recommended default for reinforcement learning, where the stochastic
    policy gradient already introduces high variance that swamps the
    benefit of second-order corrections.

    Parameters
    ----------
    inner_lr
        Learning rate for the inner-loop SGD steps (used when no custom
        ``inner_optimizer_fn`` is provided).
    """

    create_graph: bool = False

    def __init__(self, inner_lr: float) -> None:
        self.inner_lr = inner_lr

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
        """Run k inner SGD steps on support data, return adapted state.

        Uses direct ``.data`` mutation since no graph is needed.
        Saves and restores the caller's state for full isolation.
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
        """Compute and apply the first-order outer-loop update.

        Parameters
        ----------
        query_losses
            Scalar losses evaluated on query sets at the adapted points.
        """
        if query_losses is None:
            raise ValueError(
                "FOMAML.meta_step() requires query_losses "
                "(one per adapted state)."
            )
        if len(query_losses) != len(adapted):
            raise ValueError(
                f"Expected {len(adapted)} query losses, "
                f"got {len(query_losses)}."
            )

        optimizer.zero_grad()
        total_loss = T.stack(list(query_losses)).mean()
        total_loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()
