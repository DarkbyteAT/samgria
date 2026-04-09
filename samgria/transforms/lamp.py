"""LAMP (Local-Averaging over Multiple Perturbations) rollback transform.

After the gradient descent step, injects uniform noise scaled by parameter
magnitude and accumulates a moving average.  After ``rollback_len + 1`` steps,
rolls back to the moving average.  Designed to be composed after a SAM or
ASAM transform in the pipeline.

LAMP uses the ``post_step`` hook because it operates on the post-descent
parameters, not on gradients.
"""

from collections.abc import Callable

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


__all__ = ["LAMPRollback"]


class LAMPRollback:
    """Local-Averaging over Multiple Perturbations with periodic rollback.

    Args:
        eps: Noise scale.  Controls the magnitude of uniform noise injected after
            each gradient descent step.
        rollback_len: Number of steps between rollbacks.  After ``rollback_len`` noisy
            updates are accumulated, parameters are replaced with their moving
            average.
    """

    def __init__(self, eps: float = 5e-3, rollback_len: int = 10) -> None:
        """Initialize LAMPRollback with the given noise scale and rollback length."""
        self.eps = eps
        self.rollback_len = rollback_len
        # Lazily initialised on first post_step() call — needs parameter vector size
        self.mean_params: T.Tensor | None = None
        self.rollback_step: int = 0

    def apply(
        self,
        model: nn.Module,
        loss_fn: Callable[..., T.Tensor],
        batch: tuple[T.Tensor, ...],
    ) -> None:
        """No-op — LAMP does not modify gradients before descent."""

    def post_step(
        self,
        model: nn.Module,
    ) -> None:
        """Inject noise, accumulate moving average, rollback when due."""
        params = parameters_to_vector(model.parameters())

        # Initialise state on first call
        if self.mean_params is None:
            self.mean_params = T.zeros_like(params)

        # Sample uniform noise scaled by parameter magnitude
        noise = params.abs() * T.empty_like(params).uniform_(-1.0, 1.0)
        noisy_params = params + (self.eps * F.normalize(noise, dim=0))
        vector_to_parameters(noisy_params, model.parameters())

        self.mean_params += noisy_params.detach()
        self.rollback_step += 1

        # Rollback to moving average if enough updates sampled
        if self.rollback_step > self.rollback_len:
            self.mean_params /= self.rollback_step
            vector_to_parameters(self.mean_params.detach(), model.parameters())
            self.mean_params = T.zeros_like(self.mean_params)
            self.rollback_step = 0
