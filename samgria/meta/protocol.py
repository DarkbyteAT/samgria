"""MetaOptimizer protocol — the interface for meta-learning algorithms.

Defines the two-method contract that all meta-optimizers must satisfy:

- ``adapt()`` runs k inner optimisation steps on a support set and
  returns an ``AdaptedState`` containing the adapted snapshot and
  optionally graph-connected parameters for second-order methods.
- ``meta_step()`` computes and applies the outer-loop parameter update
  from a collection of adapted states.

Inner-loop step function
~~~~~~~~~~~~~~~~~~~~~~~~

The inner loop is parameterised by an ``InnerStepFn``: a callable that
takes ``(params, grads)`` dicts and returns a new params dict.  The
default is vanilla SGD (``sgd(lr)``), but any differentiable optimizer
can be plugged in — the graph flows through as long as the step function
uses tensor operations rather than in-place mutation.

For non-differentiable optimizers (standard PyTorch Adam etc.), use
``mutation_optimizer(fn)`` which wraps them in the ``InnerStepFn``
interface via ``.data`` mutation.  This severs the second-order graph
but preserves first-order correctness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import torch as T
import torch.nn as nn
import torch.optim as optim

from samgria.state import AdaptedState, ParameterSnapshot
from samgria.transforms.protocol import GradientTransform


#: A single inner-loop optimisation step.  Takes the current parameter
#: dict and gradient dict, returns a new parameter dict.  The graph
#: flows through if the step uses tensor ops (not .data mutation).
InnerStepFn = Callable[[dict[str, T.Tensor], dict[str, T.Tensor]], dict[str, T.Tensor]]

#: Inner-loop regularisation callback.  Receives the current adapted
#: parameter dict and the base (pre-adaptation) parameter dict, returns
#: a scalar penalty added to the inner loss at each step.
InnerRegFn = Callable[[dict[str, T.Tensor], dict[str, T.Tensor]], T.Tensor]


def sgd(lr: float) -> InnerStepFn:
    """Vanilla SGD inner step: ``p_new = p - lr * grad``."""

    def step(
        params: dict[str, T.Tensor],
        grads: dict[str, T.Tensor],
    ) -> dict[str, T.Tensor]:
        return {k: params[k] - lr * grads[k] for k in params}

    return step


def mutation_optimizer(
    factory: Callable[..., optim.Optimizer],
) -> InnerStepFn:
    """Wrap a standard PyTorch optimizer as an ``InnerStepFn``.

    The wrapper uses ``.data`` mutation internally, which severs the
    second-order computation graph.  Use this as an escape hatch for
    non-differentiable optimizers (Adam, SGD with momentum, etc.).

    Create a fresh instance per ``adapt()`` call — the wrapper holds
    internal optimizer state that persists across inner steps but should
    not leak between tasks.

    Args:
        factory: A callable that receives an iterable of parameters and returns
            an ``optim.Optimizer``.  Called lazily on the first step.

    Example:
        ```python
        adapt(..., inner_step_fn=mutation_optimizer(lambda p: optim.Adam(p, lr=0.01)))
        ```
    """
    opt_holder: list[optim.Optimizer] = []
    param_holder: list[dict[str, nn.Parameter]] = []

    def step(
        params: dict[str, T.Tensor],
        grads: dict[str, T.Tensor],
    ) -> dict[str, T.Tensor]:
        # Lazy init: create nn.Parameters + optimizer on first call
        if not param_holder:
            nn_params = {k: nn.Parameter(v.detach().clone()) for k, v in params.items()}
            param_holder.append(nn_params)
            opt_holder.append(factory(nn_params.values()))
        else:
            # Sync current param values into the nn.Parameters
            nn_params = param_holder[0]
            with T.no_grad():
                for k in params:
                    nn_params[k].data.copy_(params[k].detach())

        nn_params = param_holder[0]
        opt = opt_holder[0]

        # Set grads and step
        opt.zero_grad()
        for k in nn_params:
            nn_params[k].grad = grads[k].detach()
        opt.step()

        # Return updated values (detached — first-order)
        return {k: nn_params[k].data.detach().clone() for k in nn_params}

    return step


def capture_base_params(
    model: nn.Module,
    inner_reg_fn: InnerRegFn | None,
) -> dict[str, T.Tensor] | None:
    """Snapshot base parameters for regularisation, if a reg fn is provided."""
    if inner_reg_fn is None:
        return None
    return {k: v.detach().clone() for k, v in model.named_parameters()}


@runtime_checkable
class MetaOptimizer(Protocol):
    """Protocol for meta-learning algorithms (MAML, FOMAML, Reptile, etc.).

    Implementations pair an inner-loop adaptation step with an outer-loop
    meta-update.  The caller's model and optimizer state are saved before
    inner-loop adaptation and restored before returning, so ``adapt()`` is
    side-effect-free from the caller's perspective.
    """

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
        """Run k inner optimisation steps on support data.

        Args:
            model: The model to adapt.  Its state is saved before adaptation and
                restored before returning — the caller's state is unchanged.
            optimizer: The outer-loop optimizer.  Its state is saved and restored
                alongside the model to maintain full isolation.
            loss_fn: Callable that takes ``*support`` and returns a scalar loss.
            support: Tuple of tensors (e.g. ``(x, y)``) passed to ``loss_fn``.
            inner_steps: Number of inner optimisation steps to run.
            grad_transforms: Optional sequence of ``GradientTransform`` instances applied
                after ``loss.backward()`` but before each inner step.
            inner_step_fn: Optional step function: ``(params, grads) -> new_params``.
                Defaults to ``sgd(inner_lr)``.  Use ``mutation_optimizer()``
                to wrap standard PyTorch optimizers.
            inner_reg_fn: Optional regularisation callback.  At each inner step,
                ``inner_reg_fn(current_params, base_params)`` is added to
                the task loss.

        Returns:
            Contains a detached ``ParameterSnapshot`` and optionally
            graph-connected parameters for second-order methods.
        """
        ...

    def meta_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        base_snapshot: ParameterSnapshot,
        adapted: Sequence[AdaptedState],
        query_losses: Sequence[T.Tensor] | None = None,
    ) -> None:
        """Compute and apply the outer-loop update.

        Args:
            model: The model to update in-place.
            optimizer: The outer-loop optimizer used to apply the update.
            base_snapshot: Snapshot of the pre-adaptation outer-loop state.
            adapted: One ``AdaptedState`` per task, produced by ``adapt()``.
            query_losses: Scalar losses evaluated on query sets at the adapted points.
                Required for gradient-based meta-optimizers (MAML, FOMAML);
                ignored by Reptile which uses parameter interpolation.
        """
        ...
