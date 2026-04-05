"""Adaptive Sharpness-Aware Minimisation (ASAM) gradient transform.

Like SAM, but the perturbation is scaled by parameter magnitude so that
the sharpness measure is invariant to parameter rescaling.

Reference: Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for
Scale-Invariant Learning of Deep Neural Networks" (ICML 2021).
"""

from collections.abc import Callable

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from samgria.utils import get_grad, set_grad


__all__ = ["ASAM"]


class ASAM:
    """Adaptive Sharpness-Aware Minimisation.

    The perturbation direction is ``|theta|^2 * grad``, normalised and scaled
    by ``rho``.  This makes the perturbation adaptive to parameter scale:
    larger parameters receive proportionally larger perturbations.

    Parameters
    ----------
    `rho`
        Perturbation radius.
    """

    def __init__(self, rho: float = 1e-2) -> None:
        self.rho = rho

    def apply(
        self,
        model: nn.Module,
        loss_fn: Callable[..., T.Tensor],
        batch: tuple[T.Tensor, ...],
    ) -> None:
        """Perturb parameters adaptively, recompute loss+grad, restore."""
        init_params = parameters_to_vector(model.parameters())
        grad = get_grad(model.parameters())

        # Scale gradient by squared parameter magnitude for adaptive perturbation
        scaled_grad = init_params.abs().square() * grad
        perturbation = self.rho * F.normalize(scaled_grad, dim=0)

        # Move to worst-case perturbation
        adv_params = init_params + perturbation
        vector_to_parameters(adv_params, model.parameters())

        # Zero gradients before recomputing at perturbed point
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = loss_fn(*batch)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]

        # Capture gradient at perturbed point, restore original parameters
        new_grad = get_grad(model.parameters())
        vector_to_parameters(init_params, model.parameters())
        set_grad(new_grad, model.parameters())

    def post_step(self, model: nn.Module) -> None:
        """No-op -- ASAM does not modify parameters after descent."""
