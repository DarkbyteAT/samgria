"""Reptile meta-optimizer.

Reptile's outer update is parameter interpolation rather than
gradient-based: ``theta += meta_lr * mean(theta'_i - theta)``.  This
avoids the need for query sets entirely — the outer signal comes from
the direction each task's inner loop moved the parameters.

The inner loop shares the same functional implementation as MAML and
FOMAML (``create_graph=False``), ensuring consistent behaviour for
GradientTransforms and inner-loop regularisation.

References:
    Nichols, Schulman (2018).  On First-Order Meta-Learning Algorithms.
    arXiv:1803.02999.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import vector_to_parameters

from samgria.meta.maml import functional_adapt
from samgria.meta.protocol import InnerRegFn, InnerStepFn
from samgria.state import AdaptedState, ParameterSnapshot
from samgria.transforms.protocol import GradientTransform


__all__ = ["Reptile"]


class Reptile:
    """Reptile meta-optimizer with parameter interpolation outer update.

    Args:
        inner_lr: Learning rate for inner-loop SGD steps.
        meta_lr: Interpolation step size for the outer update.  Controls how far
            outer parameters move toward the mean of adapted parameters.

    Note:
        ``meta_step()`` writes directly into the model's parameters via
        ``vector_to_parameters`` without calling ``optimizer.step()``.
        This means the outer optimizer's internal state (momentum, variance,
        step counts) is **not updated**.  The ``optimizer`` parameter exists
        for protocol conformance.  Learning rate schedulers attached to the
        optimizer will not progress.
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
        inner_step_fn: InnerStepFn | None = None,
        inner_reg_fn: InnerRegFn | None = None,
    ) -> AdaptedState:
        """Run k inner optimisation steps, return adapted state."""
        return functional_adapt(
            model,
            optimizer,
            loss_fn,
            support,
            inner_steps,
            self.inner_lr,
            self.create_graph,
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
        """Apply the Reptile outer update via parameter interpolation.

        Computes ``theta += meta_lr * mean(theta'_i - theta)`` and writes
        the result directly into the model's parameters.  ``query_losses``
        is accepted for protocol compatibility but ignored.  The
        ``optimizer`` is not stepped — see class docstring.
        """
        outer_params = base_snapshot.params
        mean_diff = T.stack([a.snapshot.params - outer_params for a in adapted]).mean(dim=0)

        new_params = outer_params + self.meta_lr * mean_diff
        vector_to_parameters(new_params, model.parameters())
