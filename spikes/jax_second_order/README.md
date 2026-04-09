# Spike: second-order correctness of `optax.contrib.sam` under nested `jax.grad`

**Card:** [ddbuqqQV](https://trello.com/c/ddbuqqQV) — Spike: verify second-order correctness of optax.contrib.sam under nested jax.grad
**Blocks:** [kxJXhxdH](https://trello.com/c/kxJXhxdH) — Port samgria to JAX + Equinox + Optax
**Status:** COMPLETE
**Verdict:** `ALL_AGREE` (scoped — see [Scope of Claim](#scope-of-claim) below)
**Package versions tested:** `jax 0.9.2`, `jaxlib 0.9.2`, `optax 0.2.8`, `numpy 2.4.4`, Python 3.12.3 on Apple Silicon CPU

## Question

Does `optax.contrib.sam` produce mathematically correct second-order gradients when used as the inner-loop optimizer of a MAML-style outer gradient, in both `opaque_mode=False` (transparent) and `opaque_mode=True` (opaque) modes?

The v1 synthesis at `research/samgria-buy-vs-build/synthesis.md` §2.3 speculated — without measurement — that opaque mode might silently drop second-order terms via `jax.lax.stop_gradient` inside `SAMState.cache`, citing the jax#13710 footgun. That speculation was the single most load-bearing claim in the whole synthesis.

## Experimental design

The test has two independent components.

### Component 1 — within-procedure AD-encoding consistency

For each sub-spike, compute the outer gradient of a MAML sinusoid-regression problem through three independent AD encodings:

- `rev_unrolled` — reverse-mode `jax.grad` through a Python for-loop inner loop
- `rev_scan` — reverse-mode `jax.grad` through a `jax.lax.scan` inner loop
- `fwd_unrolled` — forward-mode `jax.jacfwd` through the same Python for-loop

All three encodings implement the *same* mathematics. They differ in the iteration driver (Python loop vs. `lax.scan` carry-rewriting) and the autodiff mode (reverse-mode vs. forward-mode). Agreement cross-validates that JAX's AD machinery correctly differentiates the procedure; disagreement signals a bug in SAM, `lax.scan`, or nested-grad propagation.

### Component 2 — independent finite-differences oracle

The three AD encodings share the JAX autodiff substrate. If JAX itself had a systematic bias in how it propagates through `optax.contrib.sam` — unlikely but not a priori impossible — all three encodings would lie together and the within-procedure check would miss it. Finite differences is algebraically independent: it differentiates by *evaluating f*, not by tracing.

For each sub-spike, pick a random unit direction `v` in parameter space, then compare:

```
fd = (outer_loss(p + eps*v) - outer_loss(p - eps*v)) / (2*eps)
ad = ⟨jax.grad(outer_loss)(p), v⟩
```

Target agreement is bounded by `O(eps²) + O(roundoff/eps) ≈ sqrt(eps_machine) ≈ 1.5e-8` in float64. Anything substantially worse indicates a structural bug; agreement at the noise floor confirms AD correctness against an oracle that does not share JAX substrate.

### Sub-spikes

| Sub-spike    | Inner-loop adaptation procedure |
|--------------|---|
| `SGD`        | Plain hand-coded SGD. Harness sanity check — if this fails either component, the spike itself is broken. |
| `SAM_T`      | `optax.contrib.sam(sgd)` with `opaque_mode=False` |
| `SAM_O`      | `optax.contrib.sam(sgd)` with `opaque_mode=True` |
| `SAM_T_adam` | `optax.contrib.sam(adam, eps_root=1e-8)` with `opaque_mode=False` (chain-under-nested-grad stress test) |
| `SAM_O_adam` | `optax.contrib.sam(adam, eps_root=1e-8)` with `opaque_mode=True` (same) |

The `eps_root=1e-8` on Adam avoids the `sqrt(v=0)` singularity that arises when nested-grad differentiates through Adam's first step (a known JAX pitfall unrelated to SAM).

### Sweep

Eight configurations across multiple sync periods, inner step counts, and seeds:

```
  (seed, sync_period, inner_steps)
  (20260409, 2, 4)
  (20260409, 2, 6)
  (20260409, 3, 6)
  (20260410, 2, 4)
  (20260411, 3, 9)
  (20260412, 2, 8)
  (20260413, 1, 4)   # sync_period=1 edge case (every step is a sync)
  (20260414, 2, 5)   # inner_steps=5 with sync_period=2 (ends mid-cycle)
```

The last two configs exercise corner code paths: `sync_period=1` makes every step a sync step (`pick_one(first_step, params, cache)` is trivially `params` every iteration); `inner_steps=5, sync_period=2` means the adaptation ends mid-cycle with `state.cache ≠ params` and `steps_since_sync=0` on the final update, catching bugs where the final `SAMState` is not differentiated through correctly.

Each config produces `5 sub-spikes × (3 within-procedure pairs + 1 FD oracle) = 20` measurements. Across 8 configs: **160 total consistency measurements**.

### Source inspection

As a second line of defense, I inspected `optax.contrib._sam.sam` in `optax 0.2.8` directly. Neither `transparent_update_fn` nor `opaque_update_fn` contains any `jax.lax.stop_gradient`. The only non-arithmetic ops in transparent mode are `pick_one(cond, a, b) = cond*a + (1-cond)*b` on integer conditions (differentiable) and `jax.tree.map` (differentiable). In opaque mode, the adversarial loop is a Python `for i in range(sync_period - 1)` that calls `grad_fn = jax.grad(loss)` internally — under nested outer `jax.grad`, this is a genuine second-order computation, but nothing in the control flow introduces a `stop_gradient`. This source inspection is the real refutation of the v1 speculation's proposed mechanism; the numerical sweep confirms the inference.

## Results

All numerical measurements used float64 (`jax.config.update("jax_enable_x64", True)`) on a tiny MLP `(1, 10, 10, 1)` with 141 parameters.

### Component 1 — worst-case within-procedure `rel_diff` per sub-spike

| seed       | sync | steps | SGD       | SAM_T     | SAM_O     | SAM_T_adam | SAM_O_adam |
|------------|------|-------|-----------|-----------|-----------|------------|------------|
| 20260409   | 2    | 4     | 2.045e-16 | 1.882e-16 | 3.901e-16 | 8.828e-17  | 6.457e-16  |
| 20260409   | 2    | 6     | 3.344e-16 | 1.113e-15 | 3.466e-16 | 4.147e-16  | 3.151e-16  |
| 20260409   | 3    | 6     | 3.344e-16 | 2.427e-16 | 3.318e-16 | 2.651e-16  | 1.660e-16  |
| 20260410   | 2    | 4     | 2.583e-16 | 2.576e-16 | 1.872e-16 | 1.260e-16  | 6.779e-16  |
| 20260411   | 3    | 9     | 6.735e-16 | 2.525e-16 | 1.399e-15 | 1.009e-16  | 3.204e-16  |
| 20260412   | 2    | 8     | 8.449e-16 | 1.008e-15 | 6.602e-15 | 3.197e-16  | 8.051e-16  |
| 20260413   | 1    | 4     | 3.403e-16 | 1.236e-16 | 3.403e-16 | 8.793e-17  | 9.887e-17  |
| 20260414   | 2    | 5     | 1.880e-15 | 1.095e-16 | 6.123e-16 | 9.868e-17  | 2.721e-15  |

**Aggregate worst across the sweep:** every sub-spike ≤ `6.6e-15`, ~5 orders of magnitude below the strict tolerance of `1e-10` and ~11 orders of magnitude below the loose tolerance of `1e-4`.

### Component 2 — FD-oracle `rel_diff` (finite differences vs reverse-mode directional grad)

| seed       | sync | steps | SGD       | SAM_T     | SAM_O     | SAM_T_adam | SAM_O_adam |
|------------|------|-------|-----------|-----------|-----------|------------|------------|
| 20260409   | 2    | 4     | 1.423e-10 | 2.755e-10 | 1.357e-11 | 6.083e-11  | 1.960e-10  |
| 20260409   | 2    | 6     | 2.361e-12 | 1.194e-10 | 1.057e-10 | 5.273e-11  | 1.821e-10  |
| 20260409   | 3    | 6     | 2.361e-12 | 1.964e-09 | 2.808e-10 | 6.782e-11  | 2.424e-10  |
| 20260410   | 2    | 4     | 2.830e-10 | 6.450e-11 | 6.560e-11 | 1.427e-11  | 1.402e-08  |
| 20260411   | 3    | 9     | 3.882e-10 | 6.758e-11 | 4.356e-09 | 5.610e-12  | 1.558e-10  |
| 20260412   | 2    | 8     | 2.081e-09 | 5.044e-11 | 1.710e-09 | 3.500e-10  | 7.875e-10  |
| 20260413   | 1    | 4     | 1.078e-09 | 1.862e-11 | 1.949e-10 | 3.192e-10  | 2.111e-10  |
| 20260414   | 2    | 5     | 6.198e-09 | 1.194e-09 | 8.392e-09 | 1.124e-10  | 9.351e-09  |

**Aggregate worst across the sweep:** every sub-spike ≤ `1.4e-8`, consistent with the expected float64 FD noise floor of `sqrt(eps_machine) ≈ 1.5e-8`. No measurement approaches the loose FD tolerance of `1e-4` by 4+ orders of magnitude.

### Tolerance policy

| Gate | Tolerance | Applies to | Rationale |
|---|---|---|---|
| Within-procedure strict | `1e-10` | SGD sub-spike | Different implementations of identical math — any disagreement is a harness bug |
| Within-procedure loose  | `1e-4`  | SAM sub-spikes | `lax.scan` may rewrite accumulation order vs. the Python loop; tiny rounding drift is acceptable |
| FD oracle loose         | `1e-4`  | All sub-spikes | FD has `O(eps²) + O(roundoff/eps)` noise that bounds it at `~sqrt(eps_machine)`; we want to catch order-of-magnitude bugs, not FD noise |

All four gates pass across all 8 sweep configs for all 5 sub-spikes.

## Verdict

**`ALL_AGREE`** (scoped).

Under nested `jax.grad`, `optax.contrib.sam` produces the same second-order outer gradient via every AD path tested and agrees with an independent finite-difference oracle to within the expected noise floor. The v1 synthesis's speculation that opaque mode silently drops second-order terms is **falsified** by both source inspection and numerical measurement.

### Scope of claim

The verdict applies to `optax.contrib.sam` **as tested in the following envelope:**

- **Versions:** `jax 0.9.2`, `jaxlib 0.9.2`, `optax 0.2.8`, float64, Apple Silicon CPU.
- **Model:** small MLP (141 params). Size-independent by analysis (the computation is pure arithmetic on a small pytree; no path would activate only at scale), but not *measured* at larger sizes.
- **Adaptation depth:** `inner_steps ∈ {4, 5, 6, 8, 9}`.
- **Sync period:** `sync_period ∈ {1, 2, 3}`.
- **Inner optimizer:** `optax.sgd`, or `optax.adam(eps_root=1e-8)`. Raw `adam` (with `eps_root=0`) was not tested and is known to hit a `sqrt(v=0)` singularity under nested grad on its first step.
- **Loss:** MSE (sinusoid regression).
- **`reset_state=True`** (the default). `reset_state=False` was not tested.
- **Single-step outer gradient:** one `jax.grad(outer_loss)(params)` call, not a full training trajectory.

**Not in scope:** mixed precision, GPU/TPU accumulation order, non-MSE losses, dynamically-sized inner loops, `sync_period > 3`, `reset_state=False`, long meta-training trajectories. None are likely to change the qualitative verdict — they do not touch new code paths that would plausibly introduce a second-order-dropping bug — but they are not certified.

The companion source inspection of `optax 0.2.8 _sam.py` carries the weight for generalization: the SAM update rule is pure arithmetic plus integer-conditioned `pick_one`, and there is no `stop_gradient` in either mode. If a future optax release introduces `stop_gradient` into SAM, the verdict needs re-verification.

## Action mapping

Under the v1 synthesis's framing, this verdict affects **one** of the library's four value-prop pillars:

| Pillar | Status after spike |
|---|---|
| SAM second-order correctness under nested grad | **Resolved** — no correctness layer needed |
| `optax.chain` scan-safety under nested grad | **Partially resolved** — `SAM_T_adam` and `SAM_O_adam` are `optax.chain(sam(...), adam(...))` equivalents, so the chain-under-scan-under-nested-grad path was exercised and passed |
| `stop_gradient` / `lax.scan` interaction bugs ([jax#13710](https://github.com/jax-ml/jax/issues/13710)) | **Not directly tested** — but SAM doesn't use `stop_gradient` in either mode, so the class of bug the v1 synthesis was worried about doesn't apply to SAM specifically |
| Novel `test_explicit_implicit_consistency_at_convergence` (cross-backend MetaOptimizer test) | **Not tested, still novel** — this is an independent contribution that does not depend on the SAM verdict |

### What to drop

Drop the SAM-specific correctness layer from samgria-JAX: no need for a `test_sam_second_order_correctness_transparent`, `test_sam_second_order_correctness_opaque`, or a hand-rolled SAM shim. rltrain's JAX migration can use `optax.contrib.sam` directly.

### What to keep

- **`ImplicitMAML` via `optimistix.fixed_point` with default `ImplicitAdjoint`.** Still ~30-50 LoC, still a novel contribution (no JAX library ships a `MetaOptimizer` protocol with this backend), untested by this spike.
- **Cross-backend consistency test** (`test_explicit_implicit_consistency_at_convergence`). Still the v1 synthesis's single most novel contribution. Independent of the SAM verdict.
- **`ASAM` transform** (~100 LoC). Still does not exist in any JAX library. Upstreamable to `optax.contrib`.
- **`LAMPRollback` transform** (~200 LoC). Still does not exist anywhere. Upstreamable to `optax.contrib`.
- **Thin `MetaOptimizer` protocol** (~30 LoC). Required for the cross-backend consistency test to have something to hang off.

**Revised scope estimate:** ~300-400 LoC business logic (down from v1's 300-400 plus the extra correctness-layer glue) + ~400 LoC tests (down from v1's ~800 tests because the SAM correctness-layer suite is dropped). The library is still worth building — it just no longer needs to certify `optax.contrib.sam` itself.

### What NOT to do

**Do NOT cancel the full samgria-JAX port.** The spike resolves one of four pillars. Three remain. The card's decision matrix said "All five agree → cancel the full port" but the matrix was written before the v1 synthesis was corrected with the Optimistix finding; it frames the port around SAM correctness alone, which was only one leg of the stool.

## How to reproduce

```bash
cd spikes/jax_second_order
uv venv && source .venv/bin/activate
uv pip install 'jax>=0.4.30' 'jaxlib>=0.4.30' 'optax>=0.2.5' 'equinox>=0.11' 'numpy>=1.26'
python spike.py
```

Expected runtime: ~30 seconds on Apple Silicon CPU. No GPU required.

## File manifest

- `pyproject.toml` — isolated dependency set (jax, jaxlib, optax, equinox, numpy)
- `spike.py` — self-contained spike script: sub-spike builder, within-procedure comparisons, FD oracle, sweep driver, two-tier verdict
- `README.md` — this report
