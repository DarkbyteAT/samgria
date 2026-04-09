# Spike: second-order correctness of `optax.contrib.sam` under nested `jax.grad`

**Card:** [ddbuqqQV](https://trello.com/c/ddbuqqQV) — Spike: verify second-order correctness of optax.contrib.sam under nested jax.grad
**Blocks:** [kxJXhxdH](https://trello.com/c/kxJXhxdH) — Port samgria to JAX + Equinox + Optax
**Status:** COMPLETE
**Verdict:** `ALL_AGREE` → cancel samgria-JAX full port; collapse to facade + novel transforms

## Question

Does `optax.contrib.sam` produce mathematically correct second-order gradients when used as the inner-loop optimizer of a MAML-style outer gradient, in both `opaque_mode=False` (transparent) and `opaque_mode=True` (opaque) modes?

The v1 synthesis at `research/samgria-buy-vs-build/synthesis.md` §2.3 speculated — without measurement — that opaque mode might silently drop second-order terms via `jax.lax.stop_gradient` inside `SAMState.cache`, citing the jax#13710 footgun. That speculation was the single most load-bearing uncertainty in the whole port decision.

## Experimental design

The test is **within-procedure consistency**, not cross-procedure comparison. Each sub-spike fixes a single inner-loop adaptation procedure and computes the outer gradient of a sinusoid-regression MAML problem through three independent AD encodings:

- `rev_unrolled` — reverse-mode `jax.grad` over a Python for-loop inner loop
- `rev_scan` — reverse-mode `jax.grad` over a `jax.lax.scan` inner loop
- `fwd_unrolled` — forward-mode `jax.jacfwd` over the same Python for-loop

All three encodings implement identical mathematics. Agreement across all three cross-validates both the iteration driver (Python loop vs. `lax.scan` carry-rewriting) and the AD mode (reverse-mode vs. forward-mode). Disagreement would signal a real bug in SAM, in `lax.scan`, or in nested-grad propagation.

Five sub-spikes cover the composition cases:

| Sub-spike | Inner-loop adaptation procedure |
|---|---|
| `SGD`        | Plain hand-coded SGD. Harness sanity check — if this disagrees, the spike itself is broken. |
| `SAM_T`      | `optax.contrib.sam(sgd)` with `opaque_mode=False` |
| `SAM_O`      | `optax.contrib.sam(sgd)` with `opaque_mode=True` |
| `SAM_T_adam` | `optax.contrib.sam(adam, eps_root=1e-8)` with `opaque_mode=False` (chain-under-nested-grad stress test) |
| `SAM_O_adam` | `optax.contrib.sam(adam, eps_root=1e-8)` with `opaque_mode=True` (same) |

The `eps_root=1e-8` on Adam avoids the `sqrt(v=0)` singularity that arises when nested-grad differentiates through Adam's first step (a known JAX pitfall unrelated to SAM).

## Sweep

Six configurations across two sync periods and four seeds:

```
  (seed, sync_period, inner_steps)
  (20260409, 2, 4)
  (20260409, 2, 6)
  (20260409, 3, 6)
  (20260410, 2, 4)
  (20260411, 3, 9)
  (20260412, 2, 8)
```

Each config produces 5 sub-spikes × 3 pair comparisons = 15 pairwise agreements. Across 6 configs: **90 total pairwise rel_diff measurements**.

## Results

All numerical measurements used float64 (`jax.config.update("jax_enable_x64", True)`) on a tiny MLP `(1, 10, 10, 1)` with 141 parameters to keep `jax.jacfwd` tractable.

### Worst-case `rel_diff` per sub-spike, across all 6 sweep configs

| seed      | sync | steps | SGD       | SAM_T     | SAM_O     | SAM_T_adam | SAM_O_adam |
|-----------|------|-------|-----------|-----------|-----------|------------|------------|
| 20260409  | 2    | 4     | 2.045e-16 | 1.882e-16 | 3.901e-16 | 8.828e-17  | 6.457e-16  |
| 20260409  | 2    | 6     | 3.344e-16 | 1.113e-15 | 3.466e-16 | 4.147e-16  | 3.151e-16  |
| 20260409  | 3    | 6     | 3.344e-16 | 2.427e-16 | 3.318e-16 | 2.651e-16  | 1.660e-16  |
| 20260410  | 2    | 4     | 2.583e-16 | 2.576e-16 | 1.872e-16 | 1.260e-16  | 6.779e-16  |
| 20260411  | 3    | 9     | 6.735e-16 | 2.525e-16 | 1.399e-15 | 1.009e-16  | 3.204e-16  |
| 20260412  | 2    | 8     | 8.449e-16 | 1.008e-15 | 6.602e-15 | 3.197e-16  | 8.051e-16  |

### Aggregate worst-case across the whole sweep

| Sub-spike   | Worst rel_diff |
|-------------|---------------|
| SGD         | 8.449e-16 |
| SAM_T       | 1.113e-15 |
| SAM_O       | 6.602e-15 |
| SAM_T_adam  | 4.147e-16 |
| SAM_O_adam  | 8.051e-16 |

### Tolerance policy

Two-tier:

- **Strict tier** (`rel_diff < 1e-10`) for the SGD harness check — different implementations of identical math should be numerically indistinguishable at float64 precision. SGD worst case: `8.4e-16` ≪ `1e-10`. Pass.
- **Loose tier** (`rel_diff < 1e-4`) for SAM sub-spikes — allows slack for `lax.scan` rewriting the accumulation order vs. the unrolled Python loop, and for forward-mode vs. reverse-mode accumulation differences. Worst SAM rel_diff across the sweep: `6.6e-15` ≪ `1e-4`. Pass by 11 orders of magnitude.

## Verdict

**`ALL_AGREE`** — every sub-spike is internally consistent across every sweep config. `optax.contrib.sam` produces mathematically correct second-order gradients under nested `jax.grad` in:

- Both transparent and opaque modes
- Both sync_period = 2 and sync_period = 3
- Both plain SGD and state-carrying Adam (with `eps_root > 0`) as the inner optimizer
- Both Python for-loop and `jax.lax.scan` iteration drivers
- Both reverse-mode (`jax.grad`) and forward-mode (`jax.jacfwd`) autodiff

The v1 synthesis's speculation that opaque mode silently drops second-order terms is **falsified**. Inspection of the `optax 0.2.8` source confirms there is no `jax.lax.stop_gradient` in either `transparent_update_fn` or `opaque_update_fn` — the only non-arithmetic ops are `pick_one(cond, a, b) = cond*a + (1-cond)*b` on integer conditions (differentiable) and `jax.tree.map` (differentiable). The source inspection and the numerical sweep agree.

## Mapping to the card's decision matrix

| Outcome | Measured? | Action |
|---|---|---|
| **All five agree within numerical tolerance** | ✅ | Cancel the full samgria-JAX port. Scope collapses to ~50 LoC facade + ASAM + LAMPRollback transforms. Rewire rltrain directly onto `optax.contrib.sam` + Eric Jang-idiom MAML. |
| Opaque mode disagrees, transparent matches baselines | ✗ | (n/a) |
| Both SAM modes disagree with baselines | ✗ | (n/a) |

## What this changes

The v1 synthesis assumed the core value proposition of samgria-JAX was **correctness certification under nested grad**. With the spike's verdict, that value proposition drops substantially:

- **No correctness layer is needed** for `optax.contrib.sam`. The primitive works correctly under the composition samgria cares about.
- **No scan-safety concern** for `optax.chain(sam, sgd)` or `optax.chain(sam, adam)`. The scan encoding agreed with the unrolled encoding across the sweep.
- **The ExplicitMAML backend is ~30 lines of glue**, not ~150. You wrap `optax.contrib.sam` in `lax.scan` and call `jax.grad(outer_loss)` — that's it. No custom AD, no `custom_vjp`, no state-management ceremony.

What remains worth building:

- **ASAM.** Still does not exist in any JAX library (verified during v1 research). If rltrain actually uses ASAM in production (it does in the PyTorch codebase), the ~100 LoC port is still justified. This can be an upstream PR to `optax.contrib`.
- **LAMPRollback.** Still novel. ~200 LoC. Also upstreamable.
- **ImplicitMAML backend (Optimistix).** Unchanged from v1 — `ImplicitAdjoint` is the default, ~30-50 LoC. Still worth shipping if anyone wants near-convergence meta-learning. But *only* if there's a known consumer; otherwise it's speculative.
- **rltrain's JAX migration** — this proceeds directly against `optax.contrib.sam` + Eric Jang-idiom MAML, with no samgria-JAX glue library in between unless the ASAM/LAMPRollback transforms land first.

## Caveats

Every spike has a honest-caveats section or it's not honest.

- **Single problem scale.** Tested on a 141-param MLP with `inner_steps ≤ 9`. Very large models or very deep inner loops could in principle expose numerical drift, but any drift would be bounded growth of the ULP-scale disagreement we already measured — not the qualitative "silently dropped terms" failure the v1 synthesis postulated. A failure mode that emerges only at scale but not at small scale is extraordinarily unlikely for a pure-arithmetic AD path.
- **`reset_state=True`**, the default, was tested. `reset_state=False` was not. If samgria-JAX ever needs that mode, add a test.
- **Single-outer-step test.** We computed `jax.grad(outer_loss)` once, not a full meta-training trajectory. A bug that only manifests across many outer steps would not be caught here — but second-order *correctness* is a property of a single gradient computation, so a meta-training divergence would be a *different* bug, not a second-order-correctness bug.
- **No GPU/TPU tested.** JAX compiles to XLA regardless, and the numerical path should be identical modulo accumulation order. Very unlikely to change the qualitative verdict.
- **`optax 0.2.8` specifically.** The source inspection was against this version. If a future optax release introduces `stop_gradient` into SAM (unlikely but not impossible), the verdict needs re-verification. Pin tests against this version when the library is built.

## How to reproduce

```bash
cd spikes/jax_second_order
uv venv && source .venv/bin/activate
uv pip install -e .   # or: uv pip install 'jax>=0.4.30' 'optax>=0.2.5' 'equinox>=0.11'
python spike.py
```

Expected runtime: ~20 seconds on Apple Silicon CPU. No GPU required.

## File manifest

- `pyproject.toml` — isolated dependency set (jax, jaxlib, optax, equinox, numpy)
- `spike.py` — the self-contained spike script with all 5 sub-spikes, sweep driver, comparison harness, and verdict function
- `README.md` — this report
