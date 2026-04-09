"""Sharpness-Aware Minimisation (SAM) gradient transform.

Perturbs parameters in the gradient direction to find a worst-case point,
recomputes the loss there, then restores the original parameters and sets
the gradient to the one computed at the perturbed point.  This encourages
convergence to flatter minima.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization" (ICLR 2021).
"""

from collections.abc import Callable

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from samgria.utils import get_grad, set_grad


__all__ = ["SAM"]


class SAM:
    r"""Sharpness-Aware Minimisation.

    The perturbation is $\epsilon^* \approx \rho \cdot \nabla L / \|\nabla L\|$:
    parameters are moved along the normalised gradient by a radius $\rho$, then
    the loss is recomputed at the perturbed point to capture the worst-case
    gradient in a neighbourhood of the current parameters.

    Args:
        rho: Perturbation radius $\rho$.  Controls how far parameters are moved
            in the gradient direction before recomputing the loss.
    """

    def __init__(self, rho: float = 1e-2) -> None:
        self.rho = rho

    def apply(
        self,
        model: nn.Module,
        loss_fn: Callable[..., T.Tensor],
        batch: tuple[T.Tensor, ...],
    ) -> None:
        """Perturb parameters, recompute loss+grad, restore original params."""
        init_params = parameters_to_vector(model.parameters())
        init_grad = F.normalize(get_grad(model.parameters()), dim=0)

        # Move to worst-case perturbation
        adv_params = init_params + (self.rho * init_grad)
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
        """No-op — SAM does not modify parameters after descent."""
