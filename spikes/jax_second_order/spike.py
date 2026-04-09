"""Pre-flight spike for samgria-JAX port.

Does optax.contrib.sam produce mathematically correct second-order gradients
when used as the inner-loop optimizer of a MAML-style outer gradient?

## Experimental design

The test is *within-procedure consistency*, not cross-procedure comparison.
Each "sub-spike" fixes a single inner-loop adaptation procedure and computes
the outer gradient of a sinusoid-regression MAML problem through THREE
independent AD encodings. Those three should agree within float64 tolerance
for any well-behaved procedure — disagreement flags a real bug in either the
procedure's implementation or in JAX/optax under nested differentiation.

The three encodings are:

    rev_unrolled  — reverse-mode jax.grad over a Python for-loop inner loop
    rev_scan      — reverse-mode jax.grad over a jax.lax.scan inner loop
    fwd_unrolled  — forward-mode jax.jacfwd over the same Python for-loop

Sub-spikes:
    [1] SGD      — plain hand-coded SGD inner loop. Baseline + harness sanity check.
    [2] SAM_T    — optax.contrib.sam(opaque_mode=False) wrapped around SGD.
    [3] SAM_O    — optax.contrib.sam(opaque_mode=True)  wrapped around SGD.
    [4] SAM_T_adam — same as [2] but with Adam (eps_root > 0) as the SAM inner.
                     Stress-tests chain-under-nested-grad per card item 8.
    [5] SAM_O_adam — same as [3] with Adam.

For each sub-spike, we compute (rev_unrolled, rev_scan, fwd_unrolled) and
check pairwise agreement. The verdict is the conjunction of per-sub-spike
verdicts, mapped to card ddbuqqQV's decision matrix:

    All sub-spikes internally consistent → cancel samgria-JAX port full scope
    SAM_T consistent, SAM_O inconsistent  → ship transparent-only with warning
    Both SAM sub-spikes inconsistent      → full samgria-JAX port justified

Run:
    cd spikes/jax_second_order && source .venv/bin/activate && python spike.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import optax.contrib


jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# Hyperparameters.
# Tiny MLP so jax.jacfwd is cheap. We care about correctness, not scale.
# -----------------------------------------------------------------------------
SEED = 20260409
N_TASKS = 2
N_SHOTS = 5
N_QUERY = 5
INNER_LR = 1e-2
RHO = 5e-2
MLP_DIMS = (1, 10, 10, 1)

# Single-point config used when not sweeping (kept for readability of main()).
INNER_STEPS = 4  # Must be >= sync_period*2 for SAM to complete a full cycle
SYNC_PERIOD = 2

# Sweep configs: each (seed, sync_period, inner_steps). inner_steps chosen so
# SAM completes at least 2 full sync cycles in the first six, plus edge
# configs to exercise corner code paths:
#   (seed, 1, N)         — sync_period=1 (every step is a sync step; degenerate
#                          but hits a path where steps_since_sync never increments)
#   (seed, 2, 5)         — inner_steps=5, sync_period=2 → ends MID-CYCLE with
#                          state.cache != params and steps_since_sync == 0 on
#                          the 5th update; catches bugs where the final SAMState
#                          is not differentiated through correctly
SWEEP = [
    (20260409, 2, 4),
    (20260409, 2, 6),
    (20260409, 3, 6),
    (20260410, 2, 4),
    (20260411, 3, 9),
    (20260412, 2, 8),
    (20260413, 1, 4),  # sync_period=1 edge case
    (20260414, 2, 5),  # ends mid-cycle (odd inner_steps with sync_period=2)
]

Params = dict[str, dict[str, jax.Array]]
Tasks = dict[str, jax.Array]


# -----------------------------------------------------------------------------
# Model: tiny MLP as a plain dict-of-arrays pytree.
# -----------------------------------------------------------------------------
def init_mlp(key: jax.Array, dims: tuple[int, ...] = MLP_DIMS) -> Params:
    params: Params = {}
    keys = jax.random.split(key, len(dims) - 1)
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
        scale = jnp.sqrt(2.0 / din)  # He init
        params[f"layer_{i}"] = {
            "W": jax.random.normal(keys[i], (din, dout), dtype=jnp.float64) * scale,
            "b": jnp.zeros((dout,), dtype=jnp.float64),
        }
    return params


def mlp_forward(params: Params, x: jax.Array) -> jax.Array:
    h = x
    n_layers = len(params)
    for i in range(n_layers):
        layer = params[f"layer_{i}"]
        h = h @ layer["W"] + layer["b"]
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return h


def task_loss(params: Params, x: jax.Array, y: jax.Array) -> jax.Array:
    preds = mlp_forward(params, x)
    return jnp.mean((preds - y) ** 2)


# -----------------------------------------------------------------------------
# Task generator: Eric Jang-style sinusoid regression.
# -----------------------------------------------------------------------------
def make_sinusoid_tasks(
    key: jax.Array, n_tasks: int = N_TASKS, n_shots: int = N_SHOTS, n_query: int = N_QUERY
) -> Tasks:
    amp_key, phase_key, sx_key, qx_key = jax.random.split(key, 4)
    amps = jax.random.uniform(amp_key, (n_tasks,), minval=0.1, maxval=5.0, dtype=jnp.float64)
    phases = jax.random.uniform(phase_key, (n_tasks,), minval=0.0, maxval=jnp.pi, dtype=jnp.float64)
    support_x = jax.random.uniform(sx_key, (n_tasks, n_shots, 1), minval=-5.0, maxval=5.0, dtype=jnp.float64)
    query_x = jax.random.uniform(qx_key, (n_tasks, n_query, 1), minval=-5.0, maxval=5.0, dtype=jnp.float64)
    support_y = amps[:, None, None] * jnp.sin(support_x + phases[:, None, None])
    query_y = amps[:, None, None] * jnp.sin(query_x + phases[:, None, None])
    return {
        "support_x": support_x,
        "support_y": support_y,
        "query_x": query_x,
        "query_y": query_y,
    }


# -----------------------------------------------------------------------------
# Inner-loop adaptation procedures.
#
# Each procedure exposes TWO encodings:
#   * `*_unrolled(params, sx, sy)` — Python for-loop
#   * `*_scan(params, sx, sy)`     — jax.lax.scan
# Both encodings must implement the SAME mathematical operation. Disagreement
# between their outer gradients is a bug. Forward-mode (jacfwd) on the unrolled
# version gives us a third independent AD check of the same math.
# -----------------------------------------------------------------------------
def _sgd_step(p: Params, sx: jax.Array, sy: jax.Array) -> Params:
    g = jax.grad(task_loss)(p, sx, sy)
    return jax.tree.map(lambda pi, gi: pi - INNER_LR * gi, p, g)


def make_adapt_sgd_unrolled(inner_steps: int):
    def adapt_sgd_unrolled(params: Params, sx: jax.Array, sy: jax.Array) -> Params:
        p = params
        for _ in range(inner_steps):
            p = _sgd_step(p, sx, sy)
        return p

    return adapt_sgd_unrolled


def make_adapt_sgd_scan(inner_steps: int):
    def adapt_sgd_scan(params: Params, sx: jax.Array, sy: jax.Array) -> Params:
        def step(p, _):
            return _sgd_step(p, sx, sy), None

        p, _ = jax.lax.scan(step, params, xs=None, length=inner_steps)
        return p

    return adapt_sgd_scan


def _make_sam(
    opaque: bool, inner_opt: optax.GradientTransformation, sync_period: int
) -> optax.GradientTransformationExtraArgs:
    adv_opt = optax.chain(optax.contrib.normalize(), optax.sgd(RHO))
    return optax.contrib.sam(
        inner_opt,
        adv_opt,
        sync_period=sync_period,
        opaque_mode=opaque,
    )


def _sam_step_body(sam_opt, opaque: bool, sx, sy):
    """Returns a function (carry -> new_carry) applying one SAM update step.

    Used identically by both the Python for-loop and lax.scan encodings so the
    two encodings differ only in the iteration driver.
    """
    if opaque:

        def grad_fn(p: Params, _step: int) -> Params:
            del _step
            return jax.grad(task_loss)(p, sx, sy)

        def body(carry):
            p, s = carry
            g = jax.grad(task_loss)(p, sx, sy)
            updates, s_new = sam_opt.update(g, s, p, grad_fn=grad_fn)
            return optax.apply_updates(p, updates), s_new
    else:

        def body(carry):
            p, s = carry
            g = jax.grad(task_loss)(p, sx, sy)
            updates, s_new = sam_opt.update(g, s, p)
            return optax.apply_updates(p, updates), s_new

    return body


def make_adapt_sam_unrolled(*, opaque: bool, inner_opt, sync_period: int, inner_steps: int):
    def adapt_sam_unrolled(params: Params, sx: jax.Array, sy: jax.Array) -> Params:
        sam_opt = _make_sam(opaque=opaque, inner_opt=inner_opt, sync_period=sync_period)
        state = sam_opt.init(params)
        body = _sam_step_body(sam_opt, opaque, sx, sy)
        carry = (params, state)
        for _ in range(inner_steps):
            carry = body(carry)
        return carry[0]

    return adapt_sam_unrolled


def make_adapt_sam_scan(*, opaque: bool, inner_opt, sync_period: int, inner_steps: int):
    def adapt_sam_scan(params: Params, sx: jax.Array, sy: jax.Array) -> Params:
        sam_opt = _make_sam(opaque=opaque, inner_opt=inner_opt, sync_period=sync_period)
        state = sam_opt.init(params)
        body = _sam_step_body(sam_opt, opaque, sx, sy)

        def step(carry, _):
            return body(carry), None

        (final_p, _), _ = jax.lax.scan(step, (params, state), xs=None, length=inner_steps)
        return final_p

    return adapt_sam_scan


# -----------------------------------------------------------------------------
# Outer loss + AD encodings.
# -----------------------------------------------------------------------------
def make_outer_loss(adapt_fn):
    def outer_loss(params: Params, tasks: Tasks) -> jax.Array:
        def per_task(sx, sy, qx, qy):
            adapted = adapt_fn(params, sx, sy)
            return task_loss(adapted, qx, qy)

        losses = jax.vmap(per_task)(tasks["support_x"], tasks["support_y"], tasks["query_x"], tasks["query_y"])
        return jnp.mean(losses)

    return outer_loss


def reverse_mode_grad(adapt_fn, params: Params, tasks: Tasks) -> tuple[float, Params]:
    outer = make_outer_loss(adapt_fn)
    loss_val, grads = jax.value_and_grad(outer)(params, tasks)
    grads = jax.tree.map(lambda x: x.block_until_ready(), grads)
    return float(loss_val), grads


def forward_mode_grad(adapt_fn, params: Params, tasks: Tasks) -> tuple[float, Params]:
    """jax.jacfwd over a flattened params vector, then unravel back.

    Forward-mode AD is a genuinely independent code path from reverse-mode;
    agreement with reverse_mode_grad on the same adapt_fn cross-validates both.
    """
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    outer = make_outer_loss(adapt_fn)

    def outer_flat(flat_p: jax.Array) -> jax.Array:
        return outer(unravel(flat_p), tasks)

    grad_flat = jax.jacfwd(outer_flat)(flat)
    grad_flat.block_until_ready()
    return float(outer_flat(flat)), unravel(grad_flat)


def finite_difference_directional(
    adapt_fn,
    params: Params,
    tasks: Tasks,
    *,
    direction_key: jax.Array,
    eps: float = 1e-5,
) -> tuple[float, float]:
    """Independent oracle: central finite-difference vs autodiff directional grad.

    Returns (fd_value, ad_value) where both estimate the directional derivative
    of outer_loss along a random unit direction v:

        fd = (f(p + eps*v) - f(p - eps*v)) / (2*eps)
        ad = <jax.grad(outer_loss)(p), v>

    The reverse/forward/scan/unrolled AD encodings we compare elsewhere all
    share the JAX autodiff substrate. If that substrate were systematically
    biased for this computation (unlikely but not impossible), the three
    encodings would lie together. Finite differences is algebraically
    independent — it differentiates by *evaluating f*, not by tracing — so
    it's the only truly independent oracle the spike can cheaply afford.

    Limitations: (i) it's a *directional* derivative, not the full gradient —
    one scalar per direction, so we only catch signal if the bias projects
    onto the chosen direction; (ii) finite differences itself has O(eps) bias
    plus O(1/eps) roundoff, so agreement at ~1e-6 relative is the ceiling,
    not ~1e-15; (iii) we pick one direction per call — a seed sweep over
    directions would give more coverage, but one direction catches any
    sign-consistent bug.
    """
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    v = jax.random.normal(direction_key, flat.shape, dtype=flat.dtype)
    v = v / jnp.linalg.norm(v)  # unit direction

    outer = make_outer_loss(adapt_fn)

    def outer_flat(p_flat):
        return outer(unravel(p_flat), tasks)

    fd = float((outer_flat(flat + eps * v) - outer_flat(flat - eps * v)) / (2.0 * eps))

    ad_grad_flat = jax.grad(outer_flat)(flat)
    ad = float(jnp.dot(ad_grad_flat, v))
    return fd, ad


# -----------------------------------------------------------------------------
# Pairwise comparison.
# -----------------------------------------------------------------------------
@dataclass
class Encoding:
    name: str  # "rev_unrolled" | "rev_scan" | "fwd_unrolled"
    outer_loss: float
    grads: Params
    grad_norm: float


@dataclass
class PairDiff:
    a: str
    b: str
    max_abs_diff: float
    rel_diff: float  # vs max(||a||, ||b||)


@dataclass
class SubSpike:
    name: str
    description: str
    encodings: list[Encoding]
    pairs: list[PairDiff] = field(default_factory=list)
    # Finite-difference oracle cross-check: directional derivatives along one
    # random unit direction. fd_value is a central finite difference; ad_value
    # is the reverse-mode AD projection onto the same direction. Target
    # agreement is O(eps^2 + roundoff/eps) ≈ 1e-6 with eps=1e-5 in float64.
    fd_value: float = float("nan")
    ad_value: float = float("nan")
    fd_ad_rel_diff: float = float("nan")


def _grad_norm(grads: Params) -> float:
    leaves = jax.tree.leaves(grads)
    sq = sum(float(jnp.sum(x**2)) for x in leaves)
    return float(np.sqrt(sq))


def _max_abs_diff(a: Params, b: Params) -> float:
    diffs = jax.tree.map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b)
    leaves = [float(x) for x in jax.tree.leaves(diffs)]
    return float(max(leaves)) if leaves else 0.0


def _contains_nan(grads: Params) -> bool:
    return any(bool(jnp.any(jnp.isnan(x))) for x in jax.tree.leaves(grads))


def build_subspike(
    name: str,
    description: str,
    adapt_unrolled,
    adapt_scan,
    *,
    fd_direction_key: jax.Array,
    fd_eps: float = 1e-5,
) -> SubSpike:
    loss_ru, g_ru = reverse_mode_grad(adapt_unrolled, PARAMS, TASKS)
    loss_rs, g_rs = reverse_mode_grad(adapt_scan, PARAMS, TASKS)
    loss_fu, g_fu = forward_mode_grad(adapt_unrolled, PARAMS, TASKS)
    encs = [
        Encoding("rev_unrolled", loss_ru, g_ru, _grad_norm(g_ru)),
        Encoding("rev_scan", loss_rs, g_rs, _grad_norm(g_rs)),
        Encoding("fwd_unrolled", loss_fu, g_fu, _grad_norm(g_fu)),
    ]
    pairs = []
    for i in range(len(encs)):
        for j in range(i + 1, len(encs)):
            a, b = encs[i], encs[j]
            if _contains_nan(a.grads) or _contains_nan(b.grads):
                diff = float("nan")
                rel = float("nan")
            else:
                diff = _max_abs_diff(a.grads, b.grads)
                denom = max(a.grad_norm, b.grad_norm) + 1e-30
                rel = diff / denom
            pairs.append(PairDiff(a=a.name, b=b.name, max_abs_diff=diff, rel_diff=rel))

    # Independent oracle: finite differences vs reverse-mode directional grad.
    fd_value, ad_value = finite_difference_directional(
        adapt_unrolled,
        PARAMS,
        TASKS,
        direction_key=fd_direction_key,
        eps=fd_eps,
    )
    denom = max(abs(fd_value), abs(ad_value), 1e-30)
    fd_ad_rel_diff = abs(fd_value - ad_value) / denom

    return SubSpike(
        name=name,
        description=description,
        encodings=encs,
        pairs=pairs,
        fd_value=fd_value,
        ad_value=ad_value,
        fd_ad_rel_diff=fd_ad_rel_diff,
    )


def print_subspike(s: SubSpike) -> None:
    print()
    print(f"=== Sub-spike {s.name}: {s.description} ===")
    print(f"{'encoding':15s}  {'outer_loss':>13s}  {'||grad||':>13s}")
    for e in s.encodings:
        loss_str = f"{e.outer_loss:13.6e}" if not np.isnan(e.outer_loss) else "          nan"
        norm_str = f"{e.grad_norm:13.6e}" if not np.isnan(e.grad_norm) else "          nan"
        print(f"{e.name:15s}  {loss_str}  {norm_str}")
    print(f"{'pair':32s}  {'max_abs_diff':>13s}  {'rel_diff':>12s}")
    print("-" * 62)
    for p in s.pairs:
        label = f"{p.a}  vs  {p.b}"
        if np.isnan(p.max_abs_diff):
            print(f"{label:32s}  {'nan':>13s}  {'nan':>12s}")
        else:
            print(f"{label:32s}  {p.max_abs_diff:13.6e}  {p.rel_diff:12.3e}")
    # Independent oracle row.
    print(f"{'FD oracle':32s}  fd={s.fd_value:.6e}  ad={s.ad_value:.6e}  rel={s.fd_ad_rel_diff:.3e}")


# -----------------------------------------------------------------------------
# Verdict — USER-IMPLEMENTED.
# -----------------------------------------------------------------------------
# TODO(daisy): implement verdict_from_subspikes.
#
# Input: dict[str, SubSpike] with keys including at minimum:
#   "SGD"          — plain SGD inner loop (harness sanity check)
#   "SAM_T"        — SAM(sgd) with opaque_mode=False
#   "SAM_O"        — SAM(sgd) with opaque_mode=True
#   "SAM_T_adam"   — SAM(adam) with opaque_mode=False  (chain stress)
#   "SAM_O_adam"   — SAM(adam) with opaque_mode=True   (chain stress)
#
# Each sub-spike has a `pairs` list with three PairDiffs:
#   rev_unrolled vs rev_scan   — different iteration drivers, same math
#   rev_unrolled vs fwd_unrolled — different AD modes on the same computation
#   rev_scan     vs fwd_unrolled — indirect consistency
# Each PairDiff has `max_abs_diff` and `rel_diff`. NaN means one of the
# encodings returned NaN (e.g., Adam-at-v=0 singularity) — treat as "failed"
# unless the sub-spike is explicitly expected to NaN (none are).
#
# Output: (verdict_tag: str, action_text: str) mapping to the card decision
# matrix. Canonical tags:
#   "ALL_AGREE"        — every sub-spike internally consistent
#   "OPAQUE_DISAGREES" — SAM_O and/or SAM_O_adam inconsistent; SAM_T and
#                        SAM_T_adam are consistent
#   "BOTH_FAIL"        — neither transparent nor opaque are consistent
#   "HARNESS_BUG"      — SGD sub-spike itself is inconsistent (spike code
#                        is broken; no SAM verdict can be trusted)
#   "UNEXPECTED"       — anything else (e.g., transparent fails but opaque
#                        passes) — requires human review
#
# My recommendation for the tolerance policy (you can override):
#   - SGD sub-spike: strict. All pairs should be below REL_STRICT = 1e-10
#     because the three encodings implement identical math. This is the
#     harness self-check — if it fails, nothing else can be trusted.
#   - SAM sub-spikes: loose. REL_LOOSE = 1e-4 is generous but catches
#     order-of-magnitude structural errors. The SAM encodings may differ
#     slightly from each other because lax.scan rewrites the computation
#     graph (traceable ops, different accumulation order), so bitwise
#     equality is not required — only "small structured disagreement
#     consistent with rounding."
#
# Suggested constants:
REL_STRICT = 1e-10
REL_LOOSE = 1e-4
# FD oracle: O(eps^2) bias + O(roundoff/eps) noise → ~1e-6 to 1e-5 at eps=1e-5
# on well-conditioned problems. Loose tolerance catches order-of-magnitude bugs
# without tripping on honest FD quadrature noise.
REL_FD_LOOSE = 1e-4


def verdict_from_subspikes_aggregate(worst: dict[str, float], worst_fd: dict[str, float]) -> tuple[str, str]:
    """Aggregate-over-sweep verdict with independent FD oracle gate.

    Inputs:
        worst    — worst-case within-procedure rel_diff per sub-spike across
                   (rev_unrolled, rev_scan, fwd_unrolled) pairs.
        worst_fd — worst-case finite-difference vs reverse-mode directional
                   rel_diff per sub-spike. The FD oracle does not share the
                   JAX AD substrate, so it catches the one failure mode the
                   within-procedure comparison structurally cannot see: all
                   three AD encodings uniformly wrong in the same direction.

    All four gates (within-procedure strict for SGD, within-procedure loose
    for SAM sub-spikes, FD loose for every sub-spike) must pass for ALL_AGREE.
    """

    def ok_strict(name: str) -> bool:
        w = worst.get(name, float("nan"))
        return not np.isnan(w) and w < REL_STRICT

    def ok_loose(name: str) -> bool:
        w = worst.get(name, float("nan"))
        return not np.isnan(w) and w < REL_LOOSE

    def ok_fd(name: str) -> bool:
        w = worst_fd.get(name, float("nan"))
        return not np.isnan(w) and w < REL_FD_LOOSE

    if not ok_strict("SGD") or not ok_fd("SGD"):
        return (
            "HARNESS_BUG",
            f"SGD harness sanity failed. Within-proc worst rel_diff="
            f"{worst.get('SGD'):.3e} (needs < {REL_STRICT:.0e}), "
            f"FD oracle worst rel_diff={worst_fd.get('SGD'):.3e} "
            f"(needs < {REL_FD_LOOSE:.0e}). Fix the baselines before "
            f"trusting any SAM verdict.",
        )

    t_ok = ok_loose("SAM_T") and ok_loose("SAM_T_adam") and ok_fd("SAM_T") and ok_fd("SAM_T_adam")
    o_ok = ok_loose("SAM_O") and ok_loose("SAM_O_adam") and ok_fd("SAM_O") and ok_fd("SAM_O_adam")

    details = (
        f"within-proc worst: T={worst['SAM_T']:.2e}, O={worst['SAM_O']:.2e}, "
        f"T_adam={worst['SAM_T_adam']:.2e}, O_adam={worst['SAM_O_adam']:.2e}; "
        f"FD-oracle worst: T={worst_fd['SAM_T']:.2e}, O={worst_fd['SAM_O']:.2e}, "
        f"T_adam={worst_fd['SAM_T_adam']:.2e}, O_adam={worst_fd['SAM_O_adam']:.2e}"
    )

    if t_ok and o_ok:
        return (
            "ALL_AGREE",
            f"All SAM sub-spikes pass within-procedure consistency AND "
            f"finite-difference oracle agreement across the sweep. {details}. "
            f"optax.contrib.sam is second-order correct under nested jax.grad "
            f"within the tested envelope. → Drop the SAM-second-order "
            f"correctness layer from samgria-JAX scope; retain ImplicitMAML, "
            f"cross-backend consistency test, ASAM, LAMPRollback.",
        )
    if t_ok and not o_ok:
        return (
            "OPAQUE_DISAGREES",
            f"Transparent SAM passes; opaque SAM fails. {details}. "
            f"→ Ship samgria-JAX transparent-only with a documented "
            f"'why opaque is unsafe under nested grad' note.",
        )
    if not t_ok and not o_ok:
        return (
            "BOTH_FAIL",
            f"Both SAM modes failed at least one gate. {details}. "
            f"→ Full samgria-JAX SAM correctness layer justified; probable "
            f"upstream optax bug, file an issue.",
        )
    return (
        "UNEXPECTED",
        f"Transparent failed but opaque passed — surprising combination. {details}. Requires manual review.",
    )


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
# We stash (PARAMS, TASKS) at module scope so build_subspike doesn't need to
# thread them through every call. This is spike code; module globals are fine.
PARAMS: Params
TASKS: Tasks


def build_all_subspikes(seed: int, sync_period: int, inner_steps: int) -> dict[str, SubSpike]:
    """Build the 5 sub-spikes for one (seed, sync_period, inner_steps) config."""
    global PARAMS, TASKS
    key = jax.random.PRNGKey(seed)
    pk, tk, fk = jax.random.split(key, 3)
    PARAMS = init_mlp(pk)
    TASKS = make_sinusoid_tasks(tk)
    # One direction key per sub-spike, all derived from the same seed so the
    # FD comparison is deterministic across re-runs.
    dk = jax.random.split(fk, 5)

    sgd_opt = optax.sgd(INNER_LR)
    adam_opt = optax.adam(INNER_LR, eps_root=1e-8)

    def make_sam_pair(*, opaque: bool, inner_opt):
        return (
            make_adapt_sam_unrolled(
                opaque=opaque,
                inner_opt=inner_opt,
                sync_period=sync_period,
                inner_steps=inner_steps,
            ),
            make_adapt_sam_scan(
                opaque=opaque,
                inner_opt=inner_opt,
                sync_period=sync_period,
                inner_steps=inner_steps,
            ),
        )

    sam_t_sgd_u, sam_t_sgd_s = make_sam_pair(opaque=False, inner_opt=sgd_opt)
    sam_o_sgd_u, sam_o_sgd_s = make_sam_pair(opaque=True, inner_opt=sgd_opt)
    sam_t_adam_u, sam_t_adam_s = make_sam_pair(opaque=False, inner_opt=adam_opt)
    sam_o_adam_u, sam_o_adam_s = make_sam_pair(opaque=True, inner_opt=adam_opt)

    subs: dict[str, SubSpike] = {}
    subs["SGD"] = build_subspike(
        "SGD",
        "plain hand-coded SGD inner loop (harness sanity)",
        make_adapt_sgd_unrolled(inner_steps),
        make_adapt_sgd_scan(inner_steps),
        fd_direction_key=dk[0],
    )
    subs["SAM_T"] = build_subspike(
        "SAM_T",
        "optax.contrib.sam(sgd)  opaque_mode=False",
        sam_t_sgd_u,
        sam_t_sgd_s,
        fd_direction_key=dk[1],
    )
    subs["SAM_O"] = build_subspike(
        "SAM_O",
        "optax.contrib.sam(sgd)  opaque_mode=True",
        sam_o_sgd_u,
        sam_o_sgd_s,
        fd_direction_key=dk[2],
    )
    subs["SAM_T_adam"] = build_subspike(
        "SAM_T_adam",
        "optax.contrib.sam(adam, eps_root=1e-8)  opaque_mode=False",
        sam_t_adam_u,
        sam_t_adam_s,
        fd_direction_key=dk[3],
    )
    subs["SAM_O_adam"] = build_subspike(
        "SAM_O_adam",
        "optax.contrib.sam(adam, eps_root=1e-8)  opaque_mode=True",
        sam_o_adam_u,
        sam_o_adam_s,
        fd_direction_key=dk[4],
    )
    return subs


def print_sweep_summary(sweep_results: list[tuple[tuple[int, int, int], dict[str, SubSpike]]]) -> None:
    """Pretty-print the worst-case rel_diff per sub-spike across the sweep."""
    print()
    print("=== SWEEP SUMMARY: worst-case rel_diff across all pair comparisons ===")
    print(
        f"{'seed':>10s}  {'sync':>4s}  {'steps':>5s}  "
        f"{'SGD':>12s}  {'SAM_T':>12s}  {'SAM_O':>12s}  "
        f"{'SAM_T_adam':>12s}  {'SAM_O_adam':>12s}"
    )
    print("-" * 98)
    for (seed, sync, steps), subs in sweep_results:
        worst = {}
        for name, s in subs.items():
            rels = [p.rel_diff for p in s.pairs]
            finite = [r for r in rels if not np.isnan(r)]
            worst[name] = max(finite) if finite else float("nan")
        print(
            f"{seed:>10d}  {sync:>4d}  {steps:>5d}  "
            f"{worst['SGD']:12.3e}  {worst['SAM_T']:12.3e}  {worst['SAM_O']:12.3e}  "
            f"{worst['SAM_T_adam']:12.3e}  {worst['SAM_O_adam']:12.3e}"
        )


def aggregate_worst(sweep_results) -> dict[str, float]:
    worst: dict[str, float] = {}
    for _, subs in sweep_results:
        for name, s in subs.items():
            for p in s.pairs:
                if np.isnan(p.rel_diff):
                    worst[name] = float("nan")
                else:
                    worst[name] = max(worst.get(name, 0.0), p.rel_diff)
    return worst


def aggregate_worst_fd(sweep_results) -> dict[str, float]:
    """Worst-case finite-difference vs AD rel_diff per sub-spike across the sweep."""
    worst: dict[str, float] = {}
    for _, subs in sweep_results:
        for name, s in subs.items():
            if np.isnan(s.fd_ad_rel_diff):
                worst[name] = float("nan")
            else:
                worst[name] = max(worst.get(name, 0.0), s.fd_ad_rel_diff)
    return worst


_SUB_KEYS = ["SGD", "SAM_T", "SAM_O", "SAM_T_adam", "SAM_O_adam"]


def print_fd_summary(sweep_results) -> None:
    """Pretty-print FD-vs-AD agreement per sub-spike across the sweep."""
    print()
    print("=== FD ORACLE SUMMARY: rel_diff of finite-differences vs reverse-mode directional grad ===")
    header = f"{'seed':>10s}  {'sync':>4s}  {'steps':>5s}  "
    header += "  ".join(f"{k:>12s}" for k in _SUB_KEYS)
    print(header)
    print("-" * 98)
    for (seed, sync, steps), subs in sweep_results:
        row = f"{seed:>10d}  {sync:>4d}  {steps:>5d}  "
        row += "  ".join(f"{subs[k].fd_ad_rel_diff:12.3e}" for k in _SUB_KEYS)
        print(row)


def main() -> None:
    print(f"Spike config: inner_lr={INNER_LR}  rho={RHO}  MLP={MLP_DIMS}")
    print(f"Tasks: {N_TASKS}  shots: {N_SHOTS}  query: {N_QUERY}")
    print(f"Sweep over {len(SWEEP)} configs: (seed, sync_period, inner_steps)")

    sweep_results = []
    for cfg in SWEEP:
        seed, sync_period, inner_steps = cfg
        print()
        print(f"--- running config seed={seed} sync={sync_period} steps={inner_steps} ---")
        subs = build_all_subspikes(seed, sync_period, inner_steps)
        sweep_results.append((cfg, subs))
        # Print the first config's detail tables for inspection; suppress rest.
        if cfg == SWEEP[0]:
            for s in subs.values():
                print_subspike(s)

    print_sweep_summary(sweep_results)
    print_fd_summary(sweep_results)
    worst = aggregate_worst(sweep_results)
    worst_fd = aggregate_worst_fd(sweep_results)

    # --- Verdict ---
    print()
    print("=== VERDICT ===")
    try:
        tag, msg = verdict_from_subspikes_aggregate(worst, worst_fd)
        print(f"{tag}: {msg}")
    except NotImplementedError as e:
        print(f"⚠  verdict_from_subspikes_aggregate is not yet implemented: {e}")
        print("   Inspect the tables above manually, or implement the function.")


if __name__ == "__main__":
    main()
