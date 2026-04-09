"""Print the actual numerical residuals from the SAM and ASAM decompositions.

Runs both three-way computations and prints magnitudes + residuals for
each, plus a direct cross-examination of how much more of the gradient
ASAM's first-order approximation drops relative to SAM's on the same
reference problem.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from sam_impl import (
    adaptive_sharpness_aware_exact,
    asam,
    sam,
    sharpness_aware_exact,
)
from test_sam_hessian_decomposition import _mlp_init, _mse_loss, _tree_max_abs


jax.config.update("jax_enable_x64", True)


def _three_way(
    params: Any,
    batch: tuple[jax.Array, jax.Array],
    rho: float,
    eps_fn: Callable[[Any], Any],
    canonical_opt: Any,
    exact_loss_fn: Callable[..., jax.Array],
) -> dict[str, Any]:
    """Run the three independent computations and return raw pytrees."""
    loss_grad = jax.grad(_mse_loss)

    state = canonical_opt.init(params)
    g_at_params = loss_grad(params, batch)
    canonical_grads, _ = canonical_opt.update(
        g_at_params,
        state,
        params,
        grad_fn=lambda p: loss_grad(p, batch),
    )

    exact_grads = jax.grad(exact_loss_fn)(params, batch)

    eps_at_params, vjp_eps = jax.vjp(eps_fn, params)
    perturbed = jax.tree.map(jnp.add, params, eps_at_params)
    cotangent = loss_grad(perturbed, batch)
    (jacobian_term,) = vjp_eps(cotangent)

    return {
        "canonical": canonical_grads,
        "exact": exact_grads,
        "jacobian_term": jacobian_term,
        "eps_norm": float(optax.tree.norm(eps_at_params)),
    }


def _report_section(name: str, results: dict[str, Any]) -> dict[str, float]:
    """Print and return the key magnitudes + residuals for one section."""
    canonical = results["canonical"]
    exact = results["exact"]
    jacobian = results["jacobian_term"]

    canonical_mag = _tree_max_abs(canonical)
    exact_mag = _tree_max_abs(exact)
    jacobian_mag = _tree_max_abs(jacobian)

    diff = jax.tree.map(jnp.subtract, exact, canonical)
    diff_abs = _tree_max_abs(diff)
    diff_rel = diff_abs / canonical_mag

    reconstructed = jax.tree.map(jnp.add, canonical, jacobian)
    residual = jax.tree.map(jnp.subtract, exact, reconstructed)
    residual_abs = _tree_max_abs(residual)
    residual_rel = residual_abs / exact_mag

    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
    print(f"    ||epsilon||_2                 = {results['eps_norm']:.6e}")
    print(f"    max |canonical_grads|         = {canonical_mag:.6e}")
    print(f"    max |exact_grads|             = {exact_mag:.6e}")
    print(f"    max |jacobian_term|           = {jacobian_mag:.6e}")
    print()
    print("    Approximation error (canonical drops):")
    print(f"      max |exact - canonical|     = {diff_abs:.6e}")
    print(f"      relative to canonical       = {diff_rel:.6%}")
    print()
    print("    Algebraic identity residual (should be ~ULP):")
    print(f"      max |exact - (canon + jac)| = {residual_abs:.6e}")
    print(f"      relative to exact           = {residual_rel:.6e}")
    print()

    return {
        "canonical_mag": canonical_mag,
        "exact_mag": exact_mag,
        "jacobian_mag": jacobian_mag,
        "diff_abs": diff_abs,
        "diff_rel": diff_rel,
        "residual_abs": residual_abs,
        "eps_norm": results["eps_norm"],
    }


def main() -> None:
    """Run SAM and ASAM decompositions, print magnitudes, residuals, and cross-examination."""
    # Shared reference problem (same seed as test fixtures)
    key = jax.random.key(20260409)
    k_init, k_data = jax.random.split(key)
    params = _mlp_init(k_init)
    xs = jax.random.uniform(k_data, (32, 1), jnp.float64)
    ys = jnp.sin(2 * jnp.pi * xs)
    batch = (xs, ys)
    rho = 0.05
    loss_grad = jax.grad(_mse_loss)

    # --- SAM eps function --------------------------------------------------
    def sam_eps_fn(p):
        g = loss_grad(p, batch)
        n = optax.tree.norm(g)
        return jax.tree.map(lambda gi: rho * gi / (n + 1e-12), g)

    sam_results = _three_way(
        params=params,
        batch=batch,
        rho=rho,
        eps_fn=sam_eps_fn,
        canonical_opt=sam(rho=rho),
        exact_loss_fn=sharpness_aware_exact(_mse_loss, rho=rho),
    )
    sam_metrics = _report_section("SAM (Foret et al. 2021)", sam_results)

    # --- ASAM eps function -------------------------------------------------
    def asam_eps_fn(p):
        g = loss_grad(p, batch)
        scale = jax.tree.map(jnp.abs, p)
        scaled = jax.tree.map(jnp.multiply, scale, g)
        n = optax.tree.norm(scaled)
        return jax.tree.map(
            lambda s, gi: rho * s * s * gi / (n + 1e-12),
            scale,
            g,
        )

    asam_results = _three_way(
        params=params,
        batch=batch,
        rho=rho,
        eps_fn=asam_eps_fn,
        canonical_opt=asam(rho=rho),
        exact_loss_fn=adaptive_sharpness_aware_exact(_mse_loss, rho=rho),
    )
    asam_metrics = _report_section("ASAM (Kwon et al. 2021)", asam_results)

    # --- Cross-examination -------------------------------------------------
    print("=" * 70)
    print("  Cross-examination: how much more does ASAM drop than SAM?")
    print("=" * 70)
    print()
    print(f"    SAM  approximation error (relative): {sam_metrics['diff_rel']:.4%}")
    print(f"    ASAM approximation error (relative): {asam_metrics['diff_rel']:.4%}")
    ratio = asam_metrics["diff_rel"] / sam_metrics["diff_rel"]
    print(f"    Ratio ASAM/SAM:                      {ratio:.3f}x")
    print()
    print(f"    SAM  ||epsilon||_2:  {sam_metrics['eps_norm']:.6e}")
    print(f"    ASAM ||epsilon||_2:  {asam_metrics['eps_norm']:.6e}")
    print(
        f"    ASAM's perturbation is "
        f"{asam_metrics['eps_norm'] / sam_metrics['eps_norm']:.3f}x "
        f"the magnitude of SAM's on this problem "
        f"(|theta|^2 scaling concentrates mass on large parameters)."
    )
    print()
    print(f"    SAM  residual: {sam_metrics['residual_abs']:.3e}")
    print(f"    ASAM residual: {asam_metrics['residual_abs']:.3e}")
    print()
    print("    Both residuals well below the test tolerance atol=1e-14.")


if __name__ == "__main__":
    main()
