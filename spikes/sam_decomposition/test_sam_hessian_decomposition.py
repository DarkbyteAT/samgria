r"""Behavioural test for the SAM / exact-SAM mathematical decomposition.

Verifies that classical SAM (Foret et al. 2021) and exact-derivative SAM
differ by exactly the Hessian-vector-product contribution that the paper
explicitly drops in §3. The assertion is an **algebraic identity** between
three independently computed quantities:

.. math::

    \underbrace{\nabla_\theta L(\theta + \varepsilon(\theta))}_{\text{exact}}
    \;=\;
    \underbrace{\nabla L(\theta + \varepsilon)}_{\text{canonical}}
    \;+\;
    \underbrace{\left[\frac{\partial \varepsilon}{\partial \theta}\right]^{\!\top}\!
                \nabla L(\theta + \varepsilon)}_{\text{Hessian term}}

where :math:`\varepsilon(\theta) = \rho \cdot g(\theta) / \|g(\theta)\|`
and :math:`g(\theta) = \nabla L(\theta)`.

Proof-by-triangulation: three numerical quantities, each computed through
a different code path, must satisfy a fixed algebraic relationship. The
Hessian term is computed via :func:`jax.vjp` of the :math:`\varepsilon`
function alone, without ever touching :func:`sharpness_aware_exact`, so
the test is non-circular.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from sam_impl import sam, sharpness_aware_exact


jax.config.update("jax_enable_x64", True)


# --- Reference problem: sinusoid regression with a tiny MLP ----------------


def _mlp_init(key: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
    """Deterministic 3-layer MLP (1 -> 8 -> 8 -> 1) at float64."""
    keys = jax.random.split(key, 3)
    shapes = [(1, 8), (8, 8), (8, 1)]
    return [
        (
            jax.random.normal(k, s, jnp.float64) * 0.5,
            jnp.zeros(s[1], jnp.float64),
        )
        for k, s in zip(keys, shapes, strict=False)
    ]


def _mlp_apply(params: list[tuple[jax.Array, jax.Array]], x: jax.Array) -> jax.Array:
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = jnp.tanh(h)
    return h


def _mse_loss(
    params: list[tuple[jax.Array, jax.Array]],
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    xs, ys = batch
    preds = _mlp_apply(params, xs)
    return jnp.mean((preds - ys) ** 2)


def _tree_allclose(a, b, *, atol: float, rtol: float) -> bool:
    la, _ = jax.tree.flatten(a)
    lb, _ = jax.tree.flatten(b)
    return all(jnp.allclose(x, y, atol=atol, rtol=rtol) for x, y in zip(la, lb, strict=False))


def _tree_max_abs(tree) -> float:
    leaves, _ = jax.tree.flatten(tree)
    return float(max(jnp.max(jnp.abs(leaf)) for leaf in leaves))


@pytest.fixture
def reference_problem():
    """Deterministic float64 sinusoid regression problem."""
    key = jax.random.key(20260409)
    k_init, k_data = jax.random.split(key)
    params = _mlp_init(k_init)
    xs = jax.random.uniform(k_data, (32, 1), jnp.float64)
    ys = jnp.sin(2 * jnp.pi * xs)
    return params, (xs, ys), 0.05  # (params, batch, rho)


# --- Tests ----------------------------------------------------------------


def test_exact_sam_decomposes_into_canonical_sam_plus_hessian_contribution(
    reference_problem,
):
    r"""Exact SAM equals canonical SAM plus the Hessian contribution, to ULP.

    Computes three quantities through three independent code paths and
    verifies the algebraic identity

    .. math:: \text{exact} = \text{canonical} + \text{Hessian term}

    to float64 precision. None of the three paths references another, so
    agreement is a cross-validation of all three.
    """
    # Given: a tiny reference problem and a loss gradient function
    params, batch, rho = reference_problem
    loss_grad = jax.grad(_mse_loss)

    def eps_fn(p):
        g = loss_grad(p, batch)
        n = optax.tree.norm(g)
        return jax.tree.map(lambda gi: rho * gi / (n + 1e-12), g)

    # When: compute the three quantities independently

    # (1) Canonical SAM: grad_fn evaluated at (params + eps) with eps
    # pre-computed as a concrete constant (no derivative through eps).
    sam_opt = sam(rho=rho)
    sam_state = sam_opt.init(params)
    g_at_params = loss_grad(params, batch)
    canonical_grads, _ = sam_opt.update(
        g_at_params,
        sam_state,
        params,
        grad_fn=lambda p: loss_grad(p, batch),
    )

    # (2) Exact SAM: d/dtheta [L(theta + eps(theta))] via jax.grad through
    # the sharpness-aware loss. The outer jax.grad flows through eps.
    exact_loss = sharpness_aware_exact(_mse_loss, rho=rho)
    exact_grads = jax.grad(exact_loss)(params, batch)

    # (3) Hessian contribution: [d eps / d theta]^T . grad L(theta + eps),
    # computed via jax.vjp of eps alone. Non-circular — does not touch
    # sharpness_aware_exact.
    eps_at_params, vjp_eps = jax.vjp(eps_fn, params)
    perturbed = jax.tree.map(jnp.add, params, eps_at_params)
    cotangent = loss_grad(perturbed, batch)
    (hessian_term,) = vjp_eps(cotangent)

    # Then: (1) + (3) reconstructs (2) to ULP precision
    reconstructed = jax.tree.map(jnp.add, canonical_grads, hessian_term)
    residual = jax.tree.map(jnp.subtract, exact_grads, reconstructed)
    max_residual = _tree_max_abs(residual)

    # Observed residual on Apple Silicon CPU, float64, this reference
    # problem: ~1.7e-18 (sub-ULP). Tolerance set to 1e-14 absolute, 1e-12
    # relative — ~10000x headroom above the observed value, enough to
    # absorb platform variation (GPU reduction ordering, JIT accumulation
    # differences) while still being tight enough that any real
    # computational error in one of the three paths would fail.
    assert _tree_allclose(exact_grads, reconstructed, atol=1e-14, rtol=1e-12), (
        f"Algebraic identity failed: "
        f"max |exact - (canonical + hessian)| = {max_residual:.3e}. "
        f"Either canonical_grads is wrong, or exact_grads is wrong, or "
        f"the jax.vjp reconstruction of the Hessian term is wrong — but "
        f"only one can be right, and all three disagreeing simultaneously "
        f"is vanishingly unlikely."
    )


def test_canonical_sam_differs_from_exact_sam_by_a_non_trivial_margin(
    reference_problem,
):
    """Negative control — the decomposition is not vacuous.

    The first test verifies an identity, but an identity is trivially
    satisfied if both sides are zero. This test confirms that the Hessian
    contribution is measurably non-zero on the reference problem, so the
    first test's positive assertion actually constrains something.
    """
    # Given: a random-init problem far from a critical point
    params, batch, rho = reference_problem
    loss_grad = jax.grad(_mse_loss)

    # When: compute canonical and exact SAM grads
    sam_opt = sam(rho=rho)
    sam_state = sam_opt.init(params)
    canonical_grads, _ = sam_opt.update(
        loss_grad(params, batch),
        sam_state,
        params,
        grad_fn=lambda p: loss_grad(p, batch),
    )
    exact_loss = sharpness_aware_exact(_mse_loss, rho=rho)
    exact_grads = jax.grad(exact_loss)(params, batch)

    # Then: they differ by much more than ULP noise
    difference = jax.tree.map(jnp.subtract, exact_grads, canonical_grads)
    max_abs_diff = _tree_max_abs(difference)
    canonical_mag = _tree_max_abs(canonical_grads)

    assert max_abs_diff > 1e-6, (
        f"Canonical and exact SAM agreed to within {max_abs_diff:.3e} — "
        f"the test problem may have started at a critical point where "
        f"grad L is ~0 and the Hessian contribution vanishes. Pick a "
        f"different seed or a worse initialisation."
    )
    assert max_abs_diff / canonical_mag > 1e-5, (
        f"Relative difference {max_abs_diff / canonical_mag:.3e} is too small to be a meaningful negative control."
    )
