r"""Behavioural test for the ASAM / exact-ASAM mathematical decomposition.

Parallel to ``test_sam_hessian_decomposition.py`` but for ASAM.

ASAM's per-parameter perturbation

.. math::

    \varepsilon_i(\theta) = \rho \cdot \frac{|\theta_i|^2 \cdot g_i(\theta)}
                                            {\|\,|\theta| \odot g(\theta)\,\|_2}

depends on :math:`\theta` through **two** pathways: the explicit
:math:`|\theta|^2` scale, and the implicit :math:`g(\theta) = \nabla L(\theta)`.
This is strictly more structure than SAM's one-pathway dependence. The
algebraic identity is the same shape:

.. math::

    \underbrace{\nabla_\theta L(\theta + \varepsilon(\theta))}_{\text{exact}}
    \;=\;
    \underbrace{\nabla L(\theta + \varepsilon)}_{\text{canonical}}
    \;+\;
    \underbrace{\left[\frac{\partial \varepsilon}{\partial \theta}\right]^{\!\top}\!
                \nabla L(\theta + \varepsilon)}_{\text{Jacobian term}}

but the Jacobian term now includes contributions from both pathways,
computed automatically by :func:`jax.vjp` because JAX's autodiff doesn't
care how many paths the input takes to the output.

Proof-by-triangulation: same non-circular structure as the SAM test.
The ``eps_fn`` inside the test is reconstructed from first principles
and does not reference :func:`adaptive_sharpness_aware_exact`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from sam_impl import adaptive_sharpness_aware_exact, asam
from test_sam_hessian_decomposition import (
    _mse_loss,
    _tree_allclose,
    _tree_max_abs,
    reference_problem,  # re-use the same fixture
)


jax.config.update("jax_enable_x64", True)

# Re-export the fixture so pytest finds it in this module
__all__ = ["reference_problem"]


def test_exact_asam_decomposes_into_canonical_asam_plus_jacobian_contribution(
    reference_problem,
):
    r"""Exact ASAM equals canonical ASAM plus the full Jacobian contribution.

    Same structure as the SAM test, but the Jacobian term now includes
    contributions from both :math:`|\theta|^2` and :math:`\nabla L(\theta)`.
    :func:`jax.vjp` handles both pathways automatically.
    """
    # Given: the shared tiny reference problem
    params, batch, rho = reference_problem
    loss_grad = jax.grad(_mse_loss)

    def eps_fn(p):
        g = loss_grad(p, batch)
        scale = jax.tree.map(jnp.abs, p)
        scaled = jax.tree.map(jnp.multiply, scale, g)
        norm = optax.tree.norm(scaled)
        return jax.tree.map(
            lambda s, gi: rho * s * s * gi / (norm + 1e-12),
            scale,
            g,
        )

    # When: compute the three quantities independently

    # (1) Canonical ASAM
    asam_opt = asam(rho=rho)
    asam_state = asam_opt.init(params)
    g_at_params = loss_grad(params, batch)
    canonical_grads, _ = asam_opt.update(
        g_at_params,
        asam_state,
        params,
        grad_fn=lambda p: loss_grad(p, batch),
    )

    # (2) Exact ASAM
    exact_loss = adaptive_sharpness_aware_exact(_mse_loss, rho=rho)
    exact_grads = jax.grad(exact_loss)(params, batch)

    # (3) Jacobian contribution via jax.vjp of eps_fn — JAX's autodiff
    # captures BOTH pathways (|theta| scale AND g(theta) inner gradient)
    # automatically, with no explicit plumbing.
    eps_at_params, vjp_eps = jax.vjp(eps_fn, params)
    perturbed = jax.tree.map(jnp.add, params, eps_at_params)
    cotangent = loss_grad(perturbed, batch)
    (jacobian_term,) = vjp_eps(cotangent)

    # Then: (1) + (3) reconstructs (2) to ULP precision
    reconstructed = jax.tree.map(jnp.add, canonical_grads, jacobian_term)
    residual = jax.tree.map(jnp.subtract, exact_grads, reconstructed)
    max_residual = _tree_max_abs(residual)

    # Observed residual on Apple Silicon CPU, float64, this reference
    # problem: see diagnose.py output for the exact value. Tolerance set
    # to the same 1e-14 / 1e-12 as the SAM test for consistency.
    assert _tree_allclose(exact_grads, reconstructed, atol=1e-14, rtol=1e-12), (
        f"ASAM algebraic identity failed: "
        f"max |exact - (canonical + jacobian)| = {max_residual:.3e}. "
        f"Because ASAM's Jacobian term has two contributing pathways "
        f"(|theta| scale and g(theta) gradient), a bug in one pathway "
        f"would typically show up as a residual ~an order of magnitude "
        f"above the SAM test's residual — not below it."
    )


def test_canonical_asam_differs_from_exact_asam_by_a_non_trivial_margin(
    reference_problem,
):
    """Negative control — ASAM's Jacobian contribution is non-vacuous."""
    # Given: the shared tiny reference problem
    params, batch, rho = reference_problem
    loss_grad = jax.grad(_mse_loss)

    # When: compute canonical and exact ASAM grads
    asam_opt = asam(rho=rho)
    asam_state = asam_opt.init(params)
    canonical_grads, _ = asam_opt.update(
        loss_grad(params, batch),
        asam_state,
        params,
        grad_fn=lambda p: loss_grad(p, batch),
    )
    exact_loss = adaptive_sharpness_aware_exact(_mse_loss, rho=rho)
    exact_grads = jax.grad(exact_loss)(params, batch)

    # Then: they differ by more than ULP noise
    difference = jax.tree.map(jnp.subtract, exact_grads, canonical_grads)
    max_abs_diff = _tree_max_abs(difference)
    canonical_mag = _tree_max_abs(canonical_grads)

    assert max_abs_diff > 1e-6, (
        f"Canonical and exact ASAM agreed to within {max_abs_diff:.3e} — "
        f"the test problem may have degenerate scale (|theta| ~ 0 for "
        f"most parameters) or may be near a critical point. Pick a "
        f"different seed or a worse initialisation."
    )
    assert max_abs_diff / canonical_mag > 1e-5, (
        f"Relative difference {max_abs_diff / canonical_mag:.3e} is too small to be a meaningful negative control."
    )
