"""GradientTransform protocol — the interface for all gradient transforms."""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch as T
import torch.nn as nn


@runtime_checkable
class GradientTransform(Protocol):
    """A single step in the gradient transform pipeline.

    Each transform receives the model (with gradients already populated from
    ``loss.backward()``), a ``loss_fn`` callable that can recompute the loss at
    a perturbed point, and the original ``batch``.

    Two hooks are available:

    - ``apply()`` runs **before** the optimiser step — for transforms that
      modify gradients or temporarily perturb parameters (e.g. SAM, ASAM).
    - ``post_step()`` runs **after** the optimiser step — for transforms that
      operate on the updated parameters (e.g. LAMP noise injection + rollback).

    Concrete classes must implement both methods.  Use an empty body (``...``)
    for phases the transform does not participate in.
    """

    def apply(
        self,
        model: nn.Module,
        loss_fn: Callable[..., T.Tensor],
        batch: tuple[T.Tensor, ...],
    ) -> None:
        """Pre-descent hook.  Modify gradients or recompute at perturbed points."""
        ...

    def post_step(
        self,
        model: nn.Module,
    ) -> None:
        """Post-descent hook.  Operate on updated parameters."""
        ...
