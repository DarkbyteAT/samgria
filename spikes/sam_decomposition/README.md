# Spike: SAM / ASAM exact-vs-canonical algebraic decomposition

**Status:** complete. Exploratory validation for the samgria-JAX port.
**Related:** [ddbuqqQV](https://trello.com/c/ddbuqqQV) (second-order spike), [kxJXhxdH](https://trello.com/c/kxJXhxdH) (samgria JAX port).
**Parent spike:** `../jax_second_order/` — verified `optax.contrib.sam` is second-order correct under nested `jax.grad`.

## Question

Do the proposed samgria-JAX SAM and ASAM surfaces — `sam(rho)` / `asam(rho)` (canonical, first-order) and `sharpness_aware_exact(loss_fn, rho)` / `adaptive_sharpness_aware_exact(loss_fn, rho)` (exact-derivative) — satisfy the algebraic identity

$$
\nabla_\theta L(\theta + \varepsilon(\theta))
\;=\;
\underbrace{\nabla L(\theta + \varepsilon)}_{\text{canonical}}
\;+\;
\underbrace{\left[\frac{\partial \varepsilon}{\partial \theta}\right]^{\!\top}\!
            \nabla L(\theta + \varepsilon)}_{\text{Jacobian contribution}}
$$

to float64 ULP precision on a tiny reference problem? And how much does the "Jacobian contribution" actually amount to — i.e., how much does the classical first-order approximation drop?

The perturbation definitions differ:

- **SAM** (Foret et al. 2021): $\varepsilon_i(\theta) = \rho \cdot g_i / \|g\|$ — one $\theta$-pathway, through the gradient.
- **ASAM** (Kwon et al. 2021): $\varepsilon_i(\theta) = \rho \cdot |\theta_i|^2 \cdot g_i / \|\,|\theta| \odot g\,\|$ — two $\theta$-pathways, through $|\theta|$ *and* through the gradient.

## Method

Proof-by-triangulation: three independent code paths per algorithm, each computing one quantity, verified against an algebraic identity.

1. **Canonical** — via `sam(rho).update_fn(...)` / `asam(rho).update_fn(...)`.
2. **Exact** — via `jax.grad(sharpness_aware_exact(loss_fn, rho))(params, batch)` / the ASAM equivalent.
3. **Jacobian term** — via `jax.vjp(eps_fn, params)[1](grad_fn(params + eps))`, where `eps_fn` is reconstructed from first principles inside the test and does **not** reference the exact-loss functions. Non-circular cross-validation.

The tests assert `exact == canonical + jacobian_term` to `atol=1e-14, rtol=1e-12` (tightened from `1e-12/1e-10` after the first run showed the actual residual was sub-ULP). Each test is paired with a negative-control asserting that canonical and exact differ by a non-trivial margin — if both were accidentally zero, the identity would be satisfied trivially.

## Reference problem

- 3-layer MLP with tanh activations: `(1, 8, 8, 1)`, 97 parameters
- Sinusoid regression: `y = sin(2 pi x)` for `x in [0, 1]`
- Batch size 32, MSE loss, float64
- Deterministic seed: `20260409`
- `rho = 0.05`

## Results — headline

All four tests pass. Concrete numbers from `diagnose.py`:

| Quantity | SAM | ASAM |
|---|---|---|
| $\|\varepsilon\|_2$ | `5.00e-02` | `3.44e-02` |
| `max |canonical_grads|` | `8.88e-01` | `4.98e-01` |
| `max |exact_grads|` | `1.02e+00` | `4.98e-01` |
| `max |jacobian_term|` | `1.27e-01` | `2.84e-03` |
| **First-order approximation error (relative)** | **14.34%** | **0.57%** |
| Residual `|exact − (canon + jac)|` | `1.73e-18` | `6.94e-18` |

Both residuals are sub-ULP (float64 machine epsilon is ~`2.22e-16`), leaving ~10 000x tolerance headroom above the asserted `atol=1e-14`.

## Surprise finding — ASAM's approximation is *tighter*, not looser

Before running the ASAM half of the spike, we predicted — in a docstring that has since been corrected — that ASAM's first-order approximation would drop *more* than SAM's, reasoning that "ASAM's $\varepsilon$ has two $\theta$-pathways instead of one, so its Jacobian contribution has more structure." The measured numbers flatly contradict this: **ASAM drops 0.57% of the gradient magnitude, SAM drops 14.34%. ASAM is ~25x more accurate.**

**Why the prediction was wrong.** The size of the Jacobian contribution scales with $\|\varepsilon\| \cdot \|\partial\varepsilon/\partial\theta\|$, and both factors inherit ASAM's $|\theta|^2$ scaling:

1. **Smaller $\|\varepsilon\|$.** ASAM uses only 68.8% of its $\rho$ budget in L2 norm on this problem (`3.44e-02` vs `5.00e-02`). The $|\theta|$ weighting suppresses the perturbation on small-magnitude parameters, which is most of the parameters on a random-init MLP.
2. **Concentrated support.** What perturbation ASAM *does* apply lives in a lower-dimensional subspace of "important" parameters where the loss surface is more linear. The Taylor expansion that underlies the first-order approximation converges faster on that subspace.
3. **Jacobian sensitivity inherits both.** $\partial\varepsilon_{\text{ASAM}}/\partial\theta$ contains $|\theta|^2$-weighted terms that are further suppressed where the scale is small — the "extra pathway" through $|\theta|$ contributes less, not more, than the "one pathway" through $g(\theta)$ that SAM already has.

**Theoretical connection.** This is the flip side of Kwon et al. 2021's scale-invariance argument: by weighting the perturbation budget by parameter magnitude, ASAM makes its sharpness measure invariant under parameter rescaling. A side effect is that ASAM is "gentler on unimportant parameters" — which, measured empirically, turns out to make its first-order approximation substantially more accurate than SAM's on the same $\rho$.

**Practical implications for samgria-JAX:**

- `samgria.adaptive_sharpness_aware_exact` is primarily a *diagnostic*, not an expensive production training algorithm. On random-init problems, canonical ASAM is already accurate to sub-percent; you don't get much by computing the exact form except confirmation.
- **ASAM's $\rho$ is not directly comparable to SAM's $\rho$.** A user porting configs from SAM to ASAM should expect the effective perturbation magnitude to shrink by a factor that depends on the parameter-magnitude distribution. On this reference problem the ratio is 0.688; it will vary with initialisation and training state.
- The ASAM docstring in `sam_impl.py` has been corrected to say that ASAM's first-order approximation is typically *tighter* than SAM's, with a pointer to this README for the measurement.

## Secondary observations

- **Residual accumulation ~ sqrt of op count.** SAM's residual is `1.73e-18`; ASAM's is `6.94e-18` (~4x larger). ASAM's `eps_fn` has roughly 2x more elementwise operations (an additional `abs`, an additional `scale * scale * grad`, a scaled norm), and the residual grows ~4x, consistent with the random-walk accumulation of independent ULP errors from `sqrt(2)² = 2` → observed ~4x.
- **SAM's perturbed-point gradient is larger than its starting-point gradient** (`max |exact_grads|` = 1.02 vs `max |canonical_grads|` = 0.89). SAM climbs uphill in the loss landscape by construction — the perturbed point is steeper. ASAM's perturbed-point gradient is almost identical to its starting-point gradient (0.498 vs 0.498), because ASAM's smaller, magnitude-weighted perturbation barely climbs at all on this problem.

## Reproduce

```bash
cd spikes/sam_decomposition
uv venv --python 3.11 && source .venv/bin/activate
uv pip install "jax>=0.4.30" "jaxlib>=0.4.30" "optax>=0.2.5" "pytest>=8.0"
pytest test_sam_hessian_decomposition.py test_asam_hessian_decomposition.py -v
python diagnose.py
```

Expected runtime: ~6 seconds (tests) plus ~2 seconds (diagnostic) on Apple Silicon CPU.

## File manifest

- `pyproject.toml` — isolated dependency set
- `sam_impl.py` — minimal samgria-JAX SAM+ASAM surface (~160 LoC) implementing `sam`, `sharpness_aware_exact`, `asam`, `adaptive_sharpness_aware_exact`
- `test_sam_hessian_decomposition.py` — SAM decomposition + negative control
- `test_asam_hessian_decomposition.py` — ASAM decomposition + negative control (re-uses SAM's fixture)
- `diagnose.py` — runs all three-way computations for both algorithms and prints the side-by-side cross-examination
- `README.md` — this report

## Known scope limits

- **Single reference problem.** Tested on one 97-parameter MLP, one sinusoid dataset, one $\rho$, one seed. The 25x SAM/ASAM approximation-error gap may vary across problem classes; a proper sweep over $\rho \in \{0.01, 0.05, 0.1, 0.2\}$ and a few seeds would strengthen the claim. Not run in this spike.
- **CPU only, float64 only.** GPU reduction ordering could widen the sub-ULP residuals slightly, but the chosen `atol=1e-14` leaves room.
- **No stability offset in `|θ|` for ASAM.** The `davda54/sam` reference adds `eta=1e-2` inside the `abs`; this spike uses `eta=0` for mathematical cleanness, which is safe on random init but not for weights that approach zero during training.
- **SAM only tested as `first-order approximation with stop_gradient discipline implicit in the gradient-transformation API`.** The `sharpness_aware_exact` implementation deliberately omits `jax.lax.stop_gradient` on `grads` so the outer `jax.grad` flows through — this is the "exact" variant. Canonical SAM via `samgria.sam` is structurally equivalent to "Shape A with `stop_gradient`" but implemented as `GradientTransformationExtraArgs`, not as a decorated loss. The equivalence between the two was the positive assertion of the SAM test.
