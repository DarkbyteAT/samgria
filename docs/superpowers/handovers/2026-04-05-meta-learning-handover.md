# Handover: Meta-Learning Implementation (ParameterSnapshot + MAML/FOMAML/Reptile)

## Context

We're adding meta-learning support to the samgria/rltrain ecosystem. The design was arrived at through consultation with the full engineering team (staff architect, contrarian, pragmatist, ML engineer, research intern, data scientist). The key decision: **MAML/Reptile are optimization algorithms, not RL algorithms** — they belong in samgria (torch-only) with a thin RL adapter in rltrain.

## Trello Cards

All cards are on the samgria board (`69d2790d39ed1f3a6ddd37c8`) except the last one which is on rltrain (`69cafa92b7759bad35d0801f`). Each has a Definition of Done checklist.

| Card | Board | Link | Status |
|---|---|---|---|
| ParameterSnapshot + save/restore state primitives | samgria | https://trello.com/c/v9iPNxBU | Todo |
| MetaOptimizer protocol + MAML/FOMAML/Reptile | samgria | https://trello.com/c/9g00yNZA | Todo (depends on above) |
| Few-shot classification example (sinusoid + Omniglot) | samgria | https://trello.com/c/Twn96bmQ | Todo (depends on MetaOptimizer) |
| MetaTrainer adapter + RL transfer example | rltrain | https://trello.com/c/CmzHWcxc | Todo (depends on MetaOptimizer) |

## Dependency Graph

```
[Card 1] ParameterSnapshot          ← implement first, no deps
    ↓
[Card 2] MetaOptimizer + MAML/FOMAML/Reptile
    ↓                ↓
[Card 3] Few-shot    [Card 4] MetaTrainer
examples (samgria)   + RL example (rltrain)
```

**Parallelism**: Cards 3 and 4 are independent once Card 2 merges. Dispatch them as parallel agents in separate worktrees.

## Architecture Decisions (already agreed)

### 1. ParameterSnapshot lives in samgria

`samgria/state.py` — frozen dataclass capturing the full state triple:

```python
@dataclass(frozen=True)
class ParameterSnapshot:
    params: T.Tensor              # Flattened, detached, cloned
    optim_state: dict             # Deep-copied optimizer state_dict
    buffers: dict[str, T.Tensor]  # Named module buffers (bn running stats etc.)
```

Plus `save_state(model, optimizer) -> ParameterSnapshot` and `restore_state(model, optimizer, snapshot) -> None`.

**Why frozen + detached**: Snapshots are immutable values. Multiple snapshots (one per task in MAML) must not alias. Detached tensors sever the autograd graph so snapshots don't leak gradients.

**Why buffers**: Batch norm `running_mean`/`running_var` shift between tasks. Without explicit buffer restore, you get silent performance degradation that looks like concept drift but is actually state leakage. The ML engineer flagged this as the #1 subtle bug.

**Why deep-copied optimizer state**: Adam's momentum/variance buffers must be independent copies. Without this, restoring after an inner loop silently corrupts the outer optimizer.

### 2. MetaOptimizer protocol in samgria

`samgria/meta/protocol.py` — two-phase protocol mirroring GradientTransform's apply/post_step:

- `adapt(model, optimizer, loss_fn, support, inner_steps, grad_transforms=())` → `ParameterSnapshot` — runs k inner steps, returns adapted state, restores outer state before returning
- `meta_step(model, optimizer, base_snapshot, adapted_snapshots, query_losses=None)` → `None` — computes and applies outer update

### 3. Inner loop uses SGD, outer loop uses caller's optimizer

Standard in the literature (Finn et al. 2017). The inner loop does manual SGD steps (`p -= inner_lr * p.grad`). This avoids the optimizer state problem entirely — no Adam momentum to corrupt during adaptation. The outer optimizer (typically Adam) handles the meta-gradient.

### 4. GradientTransform composes inside the inner loop

`adapt()` accepts a `grad_transforms` sequence. SAM/ASAM/LAMP apply at each inner step. This directly enables the experiment: "does robust optimization improve meta-learning?"

```python
meta = FOMAML(inner_lr=0.01)
adapted = meta.adapt(model, opt, loss_fn, support, inner_steps=5, grad_transforms=[SAM(rho=0.05)])
```

### 5. Three implementations: MAML, FOMAML, Reptile

- **MAML**: `create_graph=True` for second-order gradients through inner loop
- **FOMAML**: `create_graph=False`, first-order approximation. Default for RL (cheaper, often comparable). Research intern confirmed this is standard practice.
- **Reptile**: No query set. Outer update: `θ += meta_lr * mean(θ'_i - θ)`. Parameter interpolation, not gradient-based.

### 6. samgria examples prove decoupling

Two standalone examples with **zero rltrain/gymnasium imports**:
- `examples/sinusoid_regression.py` — MAML vs FOMAML vs Reptile on sinusoid tasks (Finn et al. 2017 benchmark). Includes SAM inner loop comparison.
- `examples/omniglot_fewshot.py` — 5-way 1-shot/5-shot with FOMAML + CNN with batch norm (proves buffer isolation works).

If these run cleanly, the samgria/rltrain boundary is validated.

### 7. rltrain MetaTrainer is a thin adapter

`rltrain/meta/trainer.py` — handles RL-specific concerns only:
- TaskDistribution protocol: `sample() -> list[TaskConfig]` (each task = MDP config)
- Collects rollouts per task via existing Agent/MDP infrastructure
- Delegates all optimization to samgria's MetaOptimizer
- Uses existing FQN config system, Callback system, xptrack integration

Example: `examples/meta_cartpole.py` — FOMAML+SAM on CartPole variants with different pole lengths/masses. Tracks via xptrack with hierarchical run structure.

## Existing Code to Know About

### samgria current state
- `samgria/transforms/protocol.py` — `GradientTransform` protocol (`apply` + `post_step`)
- `samgria/transforms/sam.py` — SAM uses `parameters_to_vector`/`vector_to_parameters` for param save/restore (ParameterSnapshot generalises this)
- `samgria/utils/grad.py` — `get_grad()`/`set_grad()` for gradient vectorisation
- Tests infrastructure exists but no test files yet (conftest.py is empty, pre-commit allows exit code 5)
- `samgria/__init__.__all__` currently exports: ASAM, GradientTransform, LAMPRollback, SAM, get_grad, set_grad

### rltrain integration points
- `rltrain/agents/agent.py` — `Agent.learn()` is the fixed template: loss → backward → transforms.apply → descend → transforms.post_step
- `rltrain/trainer.py` — `Trainer` class with Callback protocol (5 hook points)
- `rltrain/utils/builders/` — FQN config resolution. `grad_transforms` already resolved from JSON.
- PPO already does param snapshot/rollback for KL early-stopping (same pattern)

## Conventions

- Python 3.11+, PyTorch aliases: `T` for torch, `nn` for torch.nn, `F` for torch.nn.functional
- NumPy-style docstrings
- Plain `def test_*` functions, Given-When-Then structure, `@pytest.mark.unit`/`integration`/`e2e`
- `@runtime_checkable` Protocol for all interfaces
- Tool configs in dedicated files: ruff.toml, pytest.ini, pyrightconfig.json (not pyproject.toml)
- Don't wrap pytest tests in classes
- No filetree diagrams in any files

## Execution Plan

1. **Card 1 first** (ParameterSnapshot) — implement in samgria, PR, merge to main
2. **Card 2 next** (MetaOptimizer + implementations) — implement in samgria, PR, merge to main
3. **Cards 3 + 4 in parallel** after Card 2 merges:
   - Card 3: samgria examples (dispatch as agent in worktree)
   - Card 4: rltrain MetaTrainer (dispatch as agent in separate worktree)

For each card: read the Trello card description and Definition of Done checklist, tick off items as you complete them, move the card through Todo → Doing → Reviewing → Done.
