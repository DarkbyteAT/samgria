"""MetaOptimizer protocol — the interface for meta-learning algorithms.

Defines the two-method contract that all meta-optimizers must satisfy:

- ``adapt()`` runs k inner optimisation steps on a support set and
  returns an ``AdaptedState`` containing the adapted snapshot and
  optionally graph-connected parameters for second-order methods.
- ``meta_step()`` computes and applies the outer-loop parameter update
  from a collection of adapted states.

The protocol mirrors ``GradientTransform``'s apply/post_step split:
adapt is a read (produces adapted state), meta_step is a write (updates
parameters in-place).

Inner-loop optimiser
~~~~~~~~~~~~~~~~~~~~

By default the inner loop uses vanilla SGD (standard in the MAML/Reptile
literature and necessary for MAML's second-order graph tractability).
Callers can override this by passing an ``inner_optimizer_fn`` to
``adapt()`` — a factory that receives the model parameters and returns
a fresh ``torch.optim.Optimizer``.  ``save_state`` / ``restore_state``
guarantee full isolation regardless of which inner optimiser is used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, runtime_checkable

import torch as T
import torch.nn as nn
import torch.optim as optim

from samgria.state import AdaptedState, ParameterSnapshot
from samgria.transforms.protocol import GradientTransform


#: Factory that builds a fresh optimizer for the inner loop.
#: Receives an iterable of parameters, returns an Optimizer instance.
InnerOptimizerFn = Callable[[Iterator[nn.Parameter]], optim.Optimizer]

#: Inner-loop regularisation callback.  Receives the current adapted
#: parameter dict and the base (pre-adaptation) parameter dict, returns
#: a scalar penalty added to the inner loss at each step.
InnerRegFn = Callable[[dict[str, T.Tensor], dict[str, T.Tensor]], T.Tensor]


def capture_base_params(
    model: nn.Module,
    inner_reg_fn: InnerRegFn | None,
) -> dict[str, T.Tensor] | None:
    """Snapshot base parameters for regularisation, if a reg fn is provided."""
    if inner_reg_fn is None:
        return None
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def apply_inner_reg(
    loss: T.Tensor,
    model: nn.Module,
    inner_reg_fn: InnerRegFn | None,
    base_params: dict[str, T.Tensor] | None,
) -> T.Tensor:
    """Add inner-loop regularisation penalty to the loss.

    Reads current parameters from the model's stored params — correct
    for FOMAML and Reptile which mutate ``.data`` in place.  MAML uses
    its own functional params dict and calls ``inner_reg_fn`` directly.
    """
    if inner_reg_fn is None:
        return loss
    assert base_params is not None
    current: dict[str, T.Tensor] = dict(model.named_parameters())
    return loss + inner_reg_fn(current, base_params)


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
        inner_optimizer_fn: InnerOptimizerFn | None = None,
        inner_reg_fn: InnerRegFn | None = None,
    ) -> AdaptedState:
        """Run k inner optimisation steps on support data, return adapted state.

        Parameters
        ----------
        model
            The model to adapt.  Its state is saved before adaptation and
            restored before returning — the caller's state is unchanged.
        optimizer
            The outer-loop optimizer.  Its state is saved and restored
            alongside the model to maintain full isolation.
        loss_fn
            Callable that takes ``*support`` and returns a scalar loss.
        support
            Tuple of tensors (e.g. ``(x, y)``) passed to ``loss_fn``.
        inner_steps
            Number of inner optimisation steps to run.
        grad_transforms
            Optional sequence of ``GradientTransform`` instances applied
            after ``loss.backward()`` but before each inner step.
            Enables composing SAM, ASAM, etc. inside the inner loop.
        inner_optimizer_fn
            Optional factory that receives model parameters and returns a
            fresh optimizer for the inner loop.  When ``None`` (default),
            vanilla SGD at ``inner_lr`` is used.
        inner_reg_fn
            Optional regularisation callback.  At each inner step,
            ``inner_reg_fn(current_params, base_params)`` is added to
            the task loss.  Each implementation uses its own parameter
            representation (stored params for FOMAML, functional params
            dict for MAML).

        Returns
        -------
        AdaptedState
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

        Parameters
        ----------
        model
            The model to update in-place.
        optimizer
            The outer-loop optimizer used to apply the update.
        base_snapshot
            Snapshot of the pre-adaptation outer-loop state.
        adapted
            One ``AdaptedState`` per task, produced by ``adapt()``.
        query_losses
            Scalar losses evaluated on query sets at the adapted points.
            Required for gradient-based meta-optimizers (MAML, FOMAML);
            ignored by Reptile which uses parameter interpolation.
        """
        ...
