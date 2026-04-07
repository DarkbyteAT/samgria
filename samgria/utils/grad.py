"""Gradient vectorisation utilities for parameter-space operations."""

from collections.abc import Iterable

import torch as T
import torch.nn as nn


def get_grad(params: Iterable[nn.Parameter]) -> T.Tensor:
    """Return the current gradient of each parameter as a flattened vector.

    Args:
        params: An ``Iterable`` of the parameters to collect gradients from.

    Returns:
        A flattened vector of the gradients of each parameter.
    """
    # p.grad is Optional[Tensor] — None for frozen parameters.
    # Callers must ensure .backward() has been called and only
    # pass parameters that require gradients.
    return T.cat([p.grad.view(-1) for p in params])  # pyright: ignore[reportOptionalMemberAccess]


@T.no_grad()
def set_grad(grads: T.Tensor, params: Iterable[nn.Parameter]) -> None:
    """Set the gradient of each parameter to the corresponding slice of a vector.

    Args:
        grads: A flattened vector of the intended gradient for each parameter.
        params: An ``Iterable`` of parameters to update the gradients of.
    """
    i = 0

    for p in params:
        p.grad = grads[i : i + p.numel()].view(p.shape).to(p.device)
        i += p.numel()
