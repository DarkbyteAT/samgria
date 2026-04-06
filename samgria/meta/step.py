"""MetaStep builder and meta_step context manager.

Provides a high-level API for executing one outer meta-learning step.
The builder collects tasks (each with support/query sets), handles the
save/restore choreography, computes query losses, and applies the outer
update.  The context manager wraps the builder with automatic state
save on enter and outer step on exit.

Typical usage::

    with meta_step(fomaml, model, optimizer, loss_fn=loss_fn, inner_steps=5) as ms:
        for support, query in task_loader:
            ms.task(support=support, query=query)

See ``samgria.meta`` module docstring for the mathematical formalism.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch as T
import torch.nn as nn
import torch.optim as optim

from samgria.meta.protocol import InnerRegFn, InnerStepFn, MetaOptimizer
from samgria.state import AdaptedState, restore_state, save_state
from samgria.transforms.protocol import GradientTransform
from samgria.utils.functional import functional_forward


__all__ = ["MetaStep", "meta_step"]


class MetaStep:
    """Builder for one outer meta-learning step.

    Collects tasks via ``.task()``, managing the save/restore plumbing,
    query loss computation, and task weighting internally.  Call
    ``.step()`` to apply the outer update — or use the ``meta_step``
    context manager which calls it automatically on exit.

    Parameters
    ----------
    meta_optimizer
        The meta-learning algorithm (MAML, FOMAML, Reptile, etc.).
    model
        The model being meta-trained.
    optimizer
        The outer-loop optimizer.
    loss_fn
        Default loss function for tasks.  Takes ``(*support)`` or
        ``(*query)`` and returns a scalar loss.
    inner_steps
        Default number of inner-loop SGD steps per task.
    grad_transforms
        Default gradient transforms for the inner loop.
    inner_step_fn
        Default inner step function: ``(params, grads) -> new_params``.
    inner_reg_fn
        Regularisation callback added to the inner loss at each step.
        Signature: ``(current_params, base_params) -> scalar_penalty``.
    """

    def __init__(
        self,
        meta_optimizer: MetaOptimizer,
        model: nn.Module,
        optimizer: optim.Optimizer,
        *,
        loss_fn: Callable[..., T.Tensor] | None = None,
        inner_steps: int | None = None,
        grad_transforms: Sequence[GradientTransform] = (),
        inner_step_fn: InnerStepFn | None = None,
        inner_reg_fn: InnerRegFn | None = None,
    ) -> None:
        self._meta_opt = meta_optimizer
        self._model = model
        self._optimizer = optimizer
        self._default_loss_fn = loss_fn
        self._default_inner_steps = inner_steps
        self._default_grad_transforms = grad_transforms
        self._default_inner_step_fn = inner_step_fn
        self._inner_reg_fn = inner_reg_fn

        self._base_snapshot = save_state(model, optimizer)
        self._adapted: list[AdaptedState] = []
        self._query_losses: list[T.Tensor] = []
        self._weights: list[float] = []
        self._has_query_losses = False

    def task(
        self,
        support: tuple[T.Tensor, ...],
        query: tuple[T.Tensor, ...] | None = None,
        *,
        loss_fn: Callable[..., T.Tensor] | None = None,
        inner_steps: int | None = None,
        grad_transforms: Sequence[GradientTransform] | None = None,
        inner_step_fn: InnerStepFn | None = None,
        query_loss_fn: Callable[[nn.Module, AdaptedState], T.Tensor] | None = None,
        weight: float = 1.0,
    ) -> AdaptedState:
        """Adapt on one task and collect the query loss.

        Parameters
        ----------
        support
            Support set tensors passed to ``loss_fn(*support)``.
        query
            Query set tensors.  ``None`` for Reptile (no query eval).
        loss_fn
            Per-task loss function override.
        inner_steps
            Per-task inner step count override.
        grad_transforms
            Per-task gradient transform override.
        inner_step_fn
            Per-task inner optimizer override.
        query_loss_fn
            Custom query loss callback.  Receives ``(model, adapted)``
            and returns a scalar loss.  Overrides the default query
            loss computation.
        weight
            Scaling factor for this task's query loss contribution.

        Returns
        -------
        AdaptedState
            The adapted state for this task (informational).
        """
        # Resolve per-task overrides
        task_loss_fn = loss_fn or self._default_loss_fn
        task_inner_steps = inner_steps or self._default_inner_steps
        task_grad_transforms = grad_transforms if grad_transforms is not None else self._default_grad_transforms
        task_inner_opt_fn = inner_step_fn or self._default_inner_step_fn

        if task_loss_fn is None:
            raise ValueError("loss_fn must be provided either at construction or per-task.")
        if task_inner_steps is None:
            raise ValueError("inner_steps must be provided either at construction or per-task.")

        # Adapt
        result = self._meta_opt.adapt(
            self._model,
            self._optimizer,
            task_loss_fn,
            support,
            task_inner_steps,
            grad_transforms=task_grad_transforms,
            inner_step_fn=task_inner_opt_fn,
            inner_reg_fn=self._inner_reg_fn,
        )
        self._adapted.append(result)
        self._weights.append(weight)

        # Compute query loss (if query set provided)
        if query is not None:
            if query_loss_fn is not None:
                q_loss = query_loss_fn(self._model, result)
            elif result.live_params is not None:
                # MAML: route through functional_call for second-order grads
                with functional_forward(self._model, result.live_params):
                    q_loss = task_loss_fn(*query)
            else:
                # FOMAML: restore adapted state, evaluate, restore base
                restore_state(self._model, self._optimizer, result.snapshot)
                q_loss = task_loss_fn(*query)
                restore_state(self._model, self._optimizer, self._base_snapshot)

            self._query_losses.append(weight * q_loss)
            self._has_query_losses = True

        return result

    def step(self) -> None:
        """Apply the outer-loop update.

        Raises
        ------
        ValueError
            If no tasks have been collected.
        """
        if not self._adapted:
            raise ValueError("MetaStep.step() called with zero tasks. Call .task() at least once before .step().")

        # Restore base state before outer update
        restore_state(self._model, self._optimizer, self._base_snapshot)

        self._meta_opt.meta_step(
            self._model,
            self._optimizer,
            self._base_snapshot,
            self._adapted,
            query_losses=self._query_losses if self._has_query_losses else None,
        )


@contextmanager
def meta_step(
    meta_optimizer: MetaOptimizer,
    model: nn.Module,
    optimizer: optim.Optimizer,
    **kwargs: Any,
) -> Iterator[MetaStep]:
    """Context manager for one outer meta-learning step.

    Saves model/optimizer state on enter, applies the outer update on
    exit.  All keyword arguments are forwarded to ``MetaStep``.

    Usage::

        with meta_step(fomaml, model, optimizer, loss_fn=loss_fn, inner_steps=5) as ms:
            for support, query in tasks:
                ms.task(support=support, query=query)
    """
    ms = MetaStep(meta_optimizer, model, optimizer, **kwargs)
    yield ms
    ms.step()
