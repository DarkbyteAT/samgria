"""MAML and FOMAML meta-optimizers.

All algorithms share a single functional inner loop.  Each inner step
is parameterised by an ``InnerStepFn`` — a callable that takes
``(params, grads)`` dicts and returns a new params dict.  The default
is vanilla SGD (``sgd(lr)``), but any differentiable optimizer can be
plugged in.

The only difference between MAML and FOMAML is ``create_graph``:

- **MAML** (``create_graph=True``): second-order derivatives flow
  through the inner steps into the outer update.
- **FOMAML** (``create_graph=False``): drops second-order terms for
  a cheaper update that is often comparably effective.

References
----------
Finn, Abbeel, Levine (2017).  Model-Agnostic Meta-Learning for Fast
Adaptation of Deep Networks.  ICML.

Nichols, Schulman (2018).  On First-Order Meta-Learning Algorithms.
arXiv:1803.02999.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

import torch as T
import torch.nn as nn
import torch.optim as optim

from samgria.meta.protocol import InnerRegFn, InnerStepFn, capture_base_params, sgd
from samgria.state import AdaptedState, ParameterSnapshot, restore_state, save_state
from samgria.transforms.protocol import GradientTransform
from samgria.utils.functional import functional_forward


__all__ = ["FOMAML", "MAML", "functional_adapt"]


def _build_snapshot(
    params: dict[str, T.Tensor],
    optimizer: optim.Optimizer,
) -> ParameterSnapshot:
    """Build a ParameterSnapshot from a functional params dict."""
    flat = T.cat([p.detach().reshape(-1) for p in params.values()])
    return ParameterSnapshot(
        params=flat,
        numel=flat.numel(),
        optim_state=copy.deepcopy(optimizer.state_dict()),
        buffers={},
    )


def functional_adapt(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: Callable[..., T.Tensor],
    support: tuple[T.Tensor, ...],
    inner_steps: int,
    inner_lr: float,
    create_graph: bool,
    grad_transforms: Sequence[GradientTransform] = (),
    inner_step_fn: InnerStepFn | None = None,
    inner_reg_fn: InnerRegFn | None = None,
) -> AdaptedState:
    """Shared functional inner loop for all gradient-based meta-optimizers.

    One code path: compute grads via ``autograd.grad``, optionally
    bridge to GradientTransforms, then apply the step function.
    """
    outer_snapshot = save_state(model, optimizer)
    base_params = capture_base_params(model, inner_reg_fn)
    step_fn = inner_step_fn or sgd(inner_lr)

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

        param_values = list(params.values())
        grads = T.autograd.grad(loss, param_values, create_graph=create_graph)
        grad_dict = dict(zip(params.keys(), grads, strict=True))

        # Bridge grads to model.parameters() for GradientTransform compat.
        if grad_transforms:
            saved_data = {}
            with T.no_grad():
                for name, p in model.named_parameters():
                    saved_data[name] = p.data.clone()
                    p.data = params[name].detach().clone()
            for name, p in model.named_parameters():
                p.grad = grad_dict[name]

            for transform in grad_transforms:
                transform.apply(model, loss_fn, support)
            for transform in grad_transforms:
                transform.post_step(model)

            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_dict[name] = p.grad
            with T.no_grad():
                for name, p in model.named_parameters():
                    p.data = saved_data[name]

        params = step_fn(params, grad_dict)
        # Ensure params require grad for the next iteration's autograd.grad
        params = {
            k: v if v.requires_grad else v.requires_grad_(True)
            for k, v in params.items()
        }

    snapshot = _build_snapshot(params, optimizer)
    restore_state(model, optimizer, outer_snapshot)
    return AdaptedState(
        snapshot=snapshot,
        live_params=params if create_graph else None,
    )


def _gradient_meta_step(
    optimizer: optim.Optimizer,
    adapted: Sequence[AdaptedState],
    query_losses: Sequence[T.Tensor] | None,
    name: str,
) -> None:
    """Shared outer-step for MAML and FOMAML."""
    if query_losses is None:
        raise ValueError(
            f"{name}.meta_step() requires query_losses "
            f"(one per adapted state)."
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


class MAML:
    """Model-Agnostic Meta-Learning with full second-order gradients.

    Parameters
    ----------
    inner_lr
        Default learning rate for the inner-loop SGD steps.
        Ignored when ``inner_step_fn`` is provided.
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
        inner_step_fn: InnerStepFn | None = None,
        inner_reg_fn: InnerRegFn | None = None,
    ) -> AdaptedState:
        """Run k differentiable inner steps, return adapted state."""
        return functional_adapt(
            model, optimizer, loss_fn, support, inner_steps,
            self.inner_lr, self.create_graph,
            grad_transforms=grad_transforms,
            inner_step_fn=inner_step_fn,
            inner_reg_fn=inner_reg_fn,
        )

    def meta_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        base_snapshot: ParameterSnapshot,
        adapted: Sequence[AdaptedState],
        query_losses: Sequence[T.Tensor] | None = None,
    ) -> None:
        """Compute and apply the second-order outer-loop update."""
        _gradient_meta_step(optimizer, adapted, query_losses, "MAML")


class FOMAML:
    """First-Order MAML — drops second-order gradients for efficiency.

    Parameters
    ----------
    inner_lr
        Default learning rate for the inner-loop SGD steps.
        Ignored when ``inner_step_fn`` is provided.
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
        inner_step_fn: InnerStepFn | None = None,
        inner_reg_fn: InnerRegFn | None = None,
    ) -> AdaptedState:
        """Run k inner SGD steps on support data, return adapted state."""
        return functional_adapt(
            model, optimizer, loss_fn, support, inner_steps,
            self.inner_lr, self.create_graph,
            grad_transforms=grad_transforms,
            inner_step_fn=inner_step_fn,
            inner_reg_fn=inner_reg_fn,
        )

    def meta_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        base_snapshot: ParameterSnapshot,
        adapted: Sequence[AdaptedState],
        query_losses: Sequence[T.Tensor] | None = None,
    ) -> None:
        """Compute and apply the first-order outer-loop update."""
        _gradient_meta_step(optimizer, adapted, query_losses, "FOMAML")
