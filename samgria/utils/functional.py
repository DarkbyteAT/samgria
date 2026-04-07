"""Functional parameter utilities for evaluating models at alternate parameters.

Provides ``functional_forward``, a context manager that temporarily routes
``model(x)`` through ``torch.func.functional_call`` with an explicit
parameter dict — no in-place mutation, no permanent monkey-patching.

Typical usage::

    params = {k: v + perturbation for k, v in model.named_parameters()}
    with functional_forward(model, params):
        output = model(x)  # uses params, not stored parameters
    # model(x) uses stored parameters again

This is used by MAML's inner loop to keep the computation graph intact
across SGD steps, and by ``query_forward`` to evaluate query losses at
adapted parameters.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch as T
import torch.nn as nn
from torch.func import functional_call


__all__ = ["functional_forward"]


@contextmanager
def functional_forward(
    model: nn.Module,
    params: dict[str, T.Tensor],
) -> Iterator[None]:
    """Temporarily route ``model(x)`` through ``functional_call``.

    While this context manager is active, any call to ``model(x)`` is
    dispatched via ``functional_call(model, params, (x,))`` instead of
    using the module's stored parameters.

    Args:
        model: The module whose forward to redirect.
        params: Named parameter tensors to use for the forward pass.  These
            may be graph-connected (e.g. from a differentiable inner loop)
            or detached.

    Note:
        The patched forward un-patches itself before calling ``functional_call``
        to prevent infinite recursion — ``functional_call`` internally calls
        ``module(...)`` which would otherwise re-enter the patch.
    """
    original_forward = model.forward

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        # Un-patch before calling functional_call to prevent recursion:
        # functional_call internally calls module(...) which invokes
        # forward — if we're still patched, that re-enters this function.
        model.forward = original_forward  # type: ignore[method-assign]
        try:
            return functional_call(model, params, args, kwargs)
        finally:
            model.forward = patched_forward  # type: ignore[method-assign]

    model.forward = patched_forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        model.forward = original_forward  # type: ignore[method-assign]
