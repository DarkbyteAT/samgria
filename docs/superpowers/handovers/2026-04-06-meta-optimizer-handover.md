# Handover: MetaOptimizer Protocol + MAML/FOMAML/Reptile (Card 2)

## Context

You're implementing Card 2 of the meta-learning feature for samgria. Card 1 (ParameterSnapshot + save/restore state primitives) has been merged to main. This card depends on it and is now unblocked.

**Trello card:** https://trello.com/c/9g00yNZA
**Board ID:** `69d2790d39ed1f3a6ddd37c8`
**Repo:** DarkbyteAT/samgria (working directory: `/Users/ammar/Documents/Research/samgria`)

## What Card 1 Delivered

`samgria/state.py` provides three exports (already on main):

```python
@dataclass(frozen=True)
class ParameterSnapshot:
    params: T.Tensor              # Flattened, detached, cloned
    numel: int                    # For architecture validation on restore
    optim_state: dict[str, Any]   # Deep-copied optimizer state_dict
    buffers: dict[str, T.Tensor]  # Named module buffers (BN running stats etc.)

def save_state(model, optimizer) -> ParameterSnapshot
def restore_state(model, optimizer, snapshot) -> None
```

Key behaviours already tested and guaranteed:
- Snapshots are immutable (frozen dataclass, detached tensors, deep-copied optimizer state)
- `restore_state` validates parameter count and buffer keys match, raises ValueError on mismatch
- `restore_state` clears all `.grad` to None after restoring (prevents stale gradient leakage)
- `restore_state` deep-copies optimizer state on restore (snapshot stays pristine across multiple restores)
- Batch norm `running_mean`/`running_var`/`num_batches_tracked` are captured and restored
- 18 tests including MAML inner-loop and Reptile outer-update smoke tests already pass

## What to Build

### 1. MetaOptimizer Protocol

`samgria/meta/protocol.py` — `@runtime_checkable` Protocol with two methods:

```python
@runtime_checkable
class MetaOptimizer(Protocol):
    def adapt(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable[..., T.Tensor],
        support: tuple[T.Tensor, ...],
        inner_steps: int,
        grad_transforms: Sequence[GradientTransform] = (),
    ) -> ParameterSnapshot:
        """Run k inner SGD steps on support data, return adapted snapshot.

        The outer-loop model and optimizer state are saved before adaptation
        and restored before returning — the caller's state is unchanged.
        """
        ...

    def meta_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        base_snapshot: ParameterSnapshot,
        adapted_snapshots: Sequence[ParameterSnapshot],
        query_losses: Sequence[T.Tensor] | None = None,
    ) -> None:
        """Compute and apply the outer-loop update."""
        ...
```

### 2. Three Implementations

**MAML** (`samgria/meta/maml.py`):
- `adapt()`: save outer state, run k inner SGD steps with `create_graph=True` so second-order gradients flow through
- `meta_step()`: move to each adapted snapshot, compute query loss, accumulate gradients, restore to base, step outer optimizer
- Constructor takes `inner_lr: float`

**FOMAML** (`samgria/meta/maml.py` — same file, subclass or separate class):
- Identical to MAML but `create_graph=False`
- First-order approximation — cheaper, often comparable performance
- Default for RL use cases

**Reptile** (`samgria/meta/reptile.py`):
- `adapt()`: same as FOMAML (no second-order needed)
- `meta_step()`: no query losses needed. Outer update is parameter interpolation: `theta += meta_lr * mean(theta'_i - theta)`. Apply via `vector_to_parameters`
- Constructor takes `inner_lr: float` and `meta_lr: float`

### 3. Inner Loop Design

The inner loop in `adapt()` uses manual SGD steps, NOT the caller's optimizer:

```python
for step in range(inner_steps):
    loss = loss_fn(*support)
    loss.backward(create_graph=self.create_graph)

    # Apply any GradientTransforms (e.g. SAM) before stepping
    for transform in grad_transforms:
        transform.apply(model, loss_fn, support)

    # Manual SGD step
    with T.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= inner_lr * p.grad
        if not self.create_graph:
            model.zero_grad()
```

This avoids corrupting the outer optimizer's momentum/variance buffers.

### 4. File Layout

```
samgria/
  meta/
    __init__.py       # Export MetaOptimizer, MAML, FOMAML, Reptile
    protocol.py       # MetaOptimizer protocol
    maml.py           # MAML + FOMAML
    reptile.py        # Reptile
  __init__.py          # Add MetaOptimizer, MAML, FOMAML, Reptile to __all__
```

### 5. GradientTransform Composability

`adapt()` accepts a `grad_transforms` sequence. Each transform's `apply()` runs after `loss.backward()` but before the manual SGD step, and `post_step()` runs after the SGD step. This enables:

```python
meta = FOMAML(inner_lr=0.01)
adapted = meta.adapt(model, opt, loss_fn, support, inner_steps=5,
                     grad_transforms=[SAM(rho=0.05)])
```

## Definition of Done (Trello Checklist)

1. MetaOptimizer protocol with adapt() and meta_step() methods
2. MAML implementation with create_graph=True second-order gradients
3. FOMAML implementation with first-order approximation
4. Reptile implementation with parameter interpolation outer update
5. adapt() accepts grad_transforms sequence for SAM composability
6. Inner loop uses SGD, outer loop uses caller optimizer
7. Unit tests: sinusoid regression with each implementation
8. Unit test: state isolation — outer model unchanged after adapt()
9. Unit test: MAML+SAM composability in inner loop
10. Unit test: FOMAML does not create higher-order graph
11. Exported in samgria.__init__.__all__
12. CI passes (lint, typecheck, test)

## Testing Strategy

Write behaviour-driven tests (Given-When-Then, plain functions, no classes). Tests serve as documentation and regression prevention. Include:

**Unit tests:**
- Protocol conformance (isinstance check for each implementation)
- State isolation: outer model params unchanged after adapt()
- Gradient clearing: no stale gradients after adapt()
- FOMAML: verify `create_graph=False` (check leaf tensors have no grad_fn)
- MAML+SAM composability: adapt with SAM in inner loop runs without error and produces different adapted params than without SAM

**Integration smoke tests:**
- Sinusoid regression: each meta-optimizer reduces loss on held-out sinusoid after 1 adapt step (proves the algorithm actually works)
- Multi-task outer step: full meta_step with multiple tasks produces a parameter update

## Existing Code to Know

- `samgria/state.py` — `ParameterSnapshot`, `save_state`, `restore_state` (Card 1, merged)
- `samgria/transforms/protocol.py` — `GradientTransform` protocol (`apply` + `post_step`)
- `samgria/transforms/sam.py` — SAM implementation (uses `parameters_to_vector`/`vector_to_parameters`)
- `samgria/utils/grad.py` — `get_grad()`/`set_grad()` for gradient vectorisation
- `tests/test_state.py` — 18 tests including MAML and Reptile smoke tests (for reference patterns)

## Conventions

- Python 3.11+, PyTorch aliases: `T` for torch, `nn` for torch.nn, `F` for torch.nn.functional
- NumPy-style docstrings
- Plain `def test_*` functions, Given-When-Then structure, `@pytest.mark.unit`/`integration`
- `@runtime_checkable` Protocol for all interfaces
- Tool configs in dedicated files (ruff.toml, pytest.ini, pyrightconfig.json)
- Don't wrap pytest tests in classes
- No filetree diagrams in any files
- Always squash-merge PRs with `--delete-branch --repo DarkbyteAT/samgria`

## Commands

```bash
source scripts/enable-venv.sh
uv run ruff check samgria/
uv run pyright samgria/
uv run pytest tests/
```

## Workflow

1. Move the Trello card to Doing
2. Create a feature branch (`feat/meta-optimizer`)
3. Write tests first (TDD), then implement
4. Run CI (ruff, pyright, pytest)
5. Tick off Trello DoD items as you complete them
6. Create PR, run `/gemini review` until convergence, then run engineering team reviewers
7. Address all review findings before requesting merge
8. Do NOT merge without explicit human approval
