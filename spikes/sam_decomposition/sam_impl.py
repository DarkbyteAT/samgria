r"""Minimal samgria-JAX SAM / ASAM surface — just enough to run the decomposition tests.

Ships four public names:

- ``sam(rho)`` — canonical Foret et al. 2021 SAM as a
  ``GradientTransformationExtraArgs``. Shape B: pre-computed ``grads`` +
  ``grad_fn`` kwarg, returns gradient evaluated at the perturbed point.
- ``sharpness_aware_exact(loss_fn, rho)`` — exact-derivative SAM as a loss
  decorator. Shape A: relies on ``jax.grad`` to flow the outer derivative
  through the perturbation. Captures the Hessian contribution that
  canonical SAM drops (Foret et al. 2021 §3 first-order approximation).
- ``asam(rho)`` — canonical Kwon et al. 2021 ASAM. Adds the
  per-parameter :math:`|\theta|` scaling that makes adaptive sharpness
  scale-invariant under reparameterization.
- ``adaptive_sharpness_aware_exact(loss_fn, rho)`` — exact-derivative ASAM.
  Captures the full chain rule through **both** pathways of
  :math:`\varepsilon(\theta)`: the :math:`|\theta|` scale *and* the
  inner :math:`\nabla L(\theta)`. Canonical ASAM drops both.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax


def sam(rho: float) -> optax.GradientTransformationExtraArgs:
    """Canonical Foret et al. 2021 SAM as a chainable optax transform.

    The returned update function expects a ``grad_fn`` kwarg at call time
    that maps ``params -> grads`` (closing over the current batch). The
    transform evaluates the incoming ``grads`` at the current ``params``,
    computes the perturbation ``eps = rho * grads / ||grads||``, and
    returns the gradient at the perturbed point.
    """

    def init_fn(params: Any) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        grads: Any,
        state: optax.EmptyState,
        params: Any,
        *,
        grad_fn: Callable[[Any], Any],
    ) -> tuple[Any, optax.EmptyState]:
        norm = optax.tree.norm(grads)
        eps = jax.tree.map(lambda g: rho * g / (norm + 1e-12), grads)
        perturbed = jax.tree.map(jnp.add, params, eps)
        sam_grads = grad_fn(perturbed)
        return sam_grads, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def sharpness_aware_exact(loss_fn: Callable[..., jax.Array], rho: float) -> Callable[..., jax.Array]:
    r"""Exact-derivative SAM loss transformation.

    Computes the **true** gradient of the sharpness-aware loss

    .. math::

        L_{\text{SAM}}^{\text{exact}}(\theta) = L(\theta + \varepsilon(\theta))

    where :math:`\varepsilon(\theta) = \rho \cdot \nabla L(\theta) / \|\nabla L(\theta)\|`,
    including the Hessian-vector-product contribution from differentiating
    through :math:`\varepsilon`.

    **This is NOT canonical SAM.** Foret et al. 2021 §3 explicitly drops the
    second-order term for computational tractability — see :func:`sam` for
    the classical first-order algorithm. This function exists for
    diagnostic and research purposes: comparing its output to :func:`sam`'s
    output tells you how valid the first-order approximation is at any
    point in training.

    Cost is roughly 3-4x an SGD step (one forward, one backward for the
    inner grad, one forward, one backward-through-grad for the outer
    derivative). Canonical SAM is 2x.
    """

    def sam_loss(params: Any, *args: Any, **kwargs: Any) -> jax.Array:
        grads = jax.grad(loss_fn)(params, *args, **kwargs)
        norm = optax.tree.norm(grads)
        eps = jax.tree.map(lambda g: rho * g / (norm + 1e-12), grads)
        perturbed = jax.tree.map(jnp.add, params, eps)
        return loss_fn(perturbed, *args, **kwargs)

    return sam_loss


def asam(rho: float) -> optax.GradientTransformationExtraArgs:
    r"""Canonical Kwon et al. 2021 ASAM as a chainable optax transform.

    Per-parameter perturbation:

    .. math::

        \varepsilon_i(\theta) = \rho \cdot \frac{|\theta_i|^2 \cdot g_i}
                                                {\|\,|\theta| \odot g\,\|_2}
        \quad \text{where} \quad g = \nabla L(\theta).

    This is the first-order form: ``scale = |params|`` and ``grads`` are
    both treated as pre-computed constants at update time, with no
    derivative flowing through either. The matching exact variant is
    :func:`adaptive_sharpness_aware_exact`.

    Note: no stability offset is added inside ``|params|``. On random
    initialisation the probability of an exact zero parameter is
    negligible; for production use with near-zero weights the test
    fixture would need to use ``|params| + eta`` (matching the
    davda54/sam reference's ``eta = 1e-2`` default).
    """

    def init_fn(params: Any) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        grads: Any,
        state: optax.EmptyState,
        params: Any,
        *,
        grad_fn: Callable[[Any], Any],
    ) -> tuple[Any, optax.EmptyState]:
        scale = jax.tree.map(jnp.abs, params)
        scaled = jax.tree.map(jnp.multiply, scale, grads)
        norm = optax.tree.norm(scaled)
        eps = jax.tree.map(
            lambda s, g: rho * s * s * g / (norm + 1e-12),
            scale,
            grads,
        )
        perturbed = jax.tree.map(jnp.add, params, eps)
        asam_grads = grad_fn(perturbed)
        return asam_grads, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def adaptive_sharpness_aware_exact(loss_fn: Callable[..., jax.Array], rho: float) -> Callable[..., jax.Array]:
    r"""Exact-derivative ASAM loss transformation.

    Computes the **true** gradient of the adaptive sharpness-aware loss

    .. math::

        L_{\text{ASAM}}^{\text{exact}}(\theta) = L(\theta + \varepsilon(\theta))

    where :math:`\varepsilon_i(\theta) = \rho \cdot |\theta_i|^2 \cdot g_i / \|\,|\theta| \odot g\,\|_2`,
    including the chain-rule contributions from **both** pathways through
    which :math:`\theta` enters :math:`\varepsilon`: the per-parameter
    :math:`|\theta|` scale *and* the inner :math:`\nabla L(\theta)`.

    **This is NOT canonical ASAM.** Kwon et al. 2021 adopt Foret et al.'s
    first-order approximation and treat both the scale and the inner
    gradient as precomputed constants.

    Somewhat counterintuitively, ASAM's first-order approximation is
    typically *tighter* than SAM's on the same :math:`\rho`, not looser.
    On the random-init sinusoid problem in the decomposition spike,
    canonical SAM drops ~14% of the gradient magnitude relative to its
    exact form, while canonical ASAM drops ~0.6%. The gap comes from
    the :math:`|\theta|^2` scaling, which both shrinks the effective
    perturbation magnitude (:math:`\|\varepsilon_{\text{ASAM}}\| < \|\varepsilon_{\text{SAM}}\|`
    for the same :math:`\rho`) and concentrates the perturbation on a
    lower-dimensional subspace of "important" parameters where the
    Taylor expansion converges faster. The naive "ASAM has two
    :math:`\theta`-pathways into :math:`\varepsilon`, so its Jacobian
    contribution must be larger than SAM's" argument counts the wrong
    thing — the size of the dropped term scales with
    :math:`\|\varepsilon\| \cdot \|\partial\varepsilon/\partial\theta\|`,
    and both factors inherit the :math:`|\theta|^2` suppression.

    Use this function to measure the approximation error on your own
    problem before assuming it's large or small.
    """

    def asam_loss(params: Any, *args: Any, **kwargs: Any) -> jax.Array:
        grads = jax.grad(loss_fn)(params, *args, **kwargs)
        scale = jax.tree.map(jnp.abs, params)
        scaled = jax.tree.map(jnp.multiply, scale, grads)
        norm = optax.tree.norm(scaled)
        eps = jax.tree.map(
            lambda s, g: rho * s * s * g / (norm + 1e-12),
            scale,
            grads,
        )
        perturbed = jax.tree.map(jnp.add, params, eps)
        return loss_fn(perturbed, *args, **kwargs)

    return asam_loss
