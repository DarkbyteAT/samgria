# Meta-Learning Guide

samgria provides three meta-learning algorithms — MAML, FOMAML, and Reptile — with a layered API that scales from one-liner convenience to full manual control.

## The Problem

Meta-learning trains a model initialisation that adapts quickly to new tasks. Each outer step:

1. Save the current parameters (the "base" state)
2. For each task: adapt on a support set (inner loop), evaluate on a query set
3. Use the query losses to update the base parameters (outer loop)

This requires careful state management — saving/restoring parameters, isolating inner-loop changes, and handling the computation graph differently for each algorithm. samgria handles all of this.

## Quick Start

Train a model that adapts to new sinusoids from 10 examples:

```python
import torch as T
import torch.nn as nn
from samgria import FOMAML, meta_step

model = nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 1))
optimizer = T.optim.Adam(model.parameters(), lr=3e-4)
fomaml = FOMAML(inner_lr=0.01)

def loss_fn(*batch):
    x, y = batch[0], batch[1]
    return ((model(x) - y) ** 2).mean()

# One outer step — adapt on 5 tasks, update base params
with meta_step(fomaml, model, optimizer, loss_fn=loss_fn, inner_steps=5) as ms:
    for support, query in task_loader:
        ms.task(support=support, query=query)
```

Swap `FOMAML` for `MAML(inner_lr=0.01)` or `Reptile(inner_lr=0.01, meta_lr=1.0)` — the consumer code is identical.

## Choosing an Algorithm

| Algorithm | Inner loop | Outer update | When to use |
|-----------|-----------|--------------|-------------|
| **MAML** | `create_graph=True` — full second-order | Backprop through inner steps | When adaptation quality matters more than compute cost |
| **FOMAML** | `create_graph=False` — first-order | Treats adapted params as constants | Default for most cases; comparable to MAML, much cheaper |
| **Reptile** | `create_graph=False` | Parameter interpolation (no query set) | When you can't compute query losses (e.g. unsupervised) |

MAML captures how the inner loop's trajectory *changes* as the outer parameters change (Hessian-vector products). FOMAML drops this term — the outer gradient is cheaper but ignores curvature. On smooth problems the difference is small; on problems with sharp decision boundaries (classification, BCE loss), MAML can converge where FOMAML plateaus.

Reptile doesn't need query data at all — its outer update moves the base parameters toward the mean of adapted parameters. Use it when computing a query loss is impractical.

## Per-Task Customisation

Tasks don't have to be uniform. Override inner steps, gradient transforms, or weighting per task:

```python
with meta_step(fomaml, model, optimizer, loss_fn=loss_fn, inner_steps=5) as ms:
    # Easy task — 1 inner step
    ms.task(support=easy_data, query=easy_query, inner_steps=1)

    # Hard task — 20 inner steps with SAM, weighted 3x
    ms.task(support=hard_data, query=hard_query,
            inner_steps=20, grad_transforms=[SAM(rho=0.05)], weight=3.0)
```

## Inner-Loop Regularisation

L2 regularisation toward the base parameters prevents inner-loop drift — useful for stabilising Reptile and preventing overfitting to small support sets:

```python
def l2_reg(current, base):
    return 0.1 * sum(((c - b) ** 2).sum() for c, b in zip(current.values(), base.values()))

with meta_step(fomaml, model, optimizer, loss_fn=loss_fn,
               inner_steps=5, inner_reg_fn=l2_reg) as ms:
    ms.task(support=support, query=query)
```

The regularisation function receives `(current_params, base_params)` as named dicts, so you can apply per-layer logic (e.g. skip biases, different coefficients per layer).

## Custom Inner Optimizers

The inner loop defaults to vanilla SGD (`sgd(lr)`). For differentiable optimizers that preserve the computation graph:

```python
from samgria.meta.protocol import sgd

# Custom differentiable step function — the graph flows through
def momentum_sgd(lr=0.01, mu=0.9):
    velocity = {}
    def step(params, grads):
        result = {}
        for k in params:
            v = velocity.get(k, torch.zeros_like(grads[k]))
            v = mu * v + grads[k]
            velocity[k] = v
            result[k] = params[k] - lr * v
        return result
    return step

result = maml.adapt(model, opt, loss_fn, support, inner_steps=5,
                    inner_step_fn=momentum_sgd(lr=0.01))
```

For standard PyTorch optimizers (Adam, SGD with momentum), use `mutation_optimizer()` which wraps them via `.data` mutation (first-order only):

```python
from samgria.meta.protocol import mutation_optimizer

result = fomaml.adapt(model, opt, loss_fn, support, inner_steps=5,
                      inner_step_fn=mutation_optimizer(lambda p: optim.Adam(p, lr=0.01)))
```

## Using the Builder Directly

The `meta_step` context manager calls `MetaStep()` on enter and `.step()` on exit. When you need to build up tasks across code paths, or conditionally decide whether to step, use the builder:

```python
from samgria import MetaStep

ms = MetaStep(fomaml, model, optimizer, loss_fn=loss_fn, inner_steps=5)

for task in task_queue:
    if task.is_valid():
        ms.task(support=task.support, query=task.query)

if ms._adapted:  # at least one task was added
    ms.step()
```

## Using the Low-Level Primitives

The context manager and builder handle save/restore choreography for you. When that's not enough — for example, in meta-RL where query data requires rolling out the adapted policy in an environment — use `adapt()` and `meta_step()` directly:

```python
from samgria import FOMAML, save_state, restore_state, query_forward

fomaml = FOMAML(inner_lr=0.01)

# Save outer state
base = save_state(model, optimizer)
adapted_states = []
query_losses = []

for env_config in task_distribution:
    # Collect support trajectory with base policy
    support_traj = rollout(env_config, model)

    # Adapt
    result = fomaml.adapt(model, optimizer, rl_loss_fn, support_traj, inner_steps=5)
    adapted_states.append(result)

    # Collect query trajectory with ADAPTED policy
    # (This is why the context manager doesn't work — you need
    # to interact with the environment between adapt and query eval)
    restore_state(model, optimizer, result.snapshot)
    query_traj = rollout(env_config, model)
    query_losses.append(rl_loss_fn(*query_traj))

    # Restore base for next task
    restore_state(model, optimizer, base)

# Outer update
fomaml.meta_step(model, optimizer, base, adapted_states, query_losses=query_losses)
```

This is the pattern the meta-RL adapter in rltrain will use. The primitives give you full control over when state is saved/restored and how query data is collected.

## Mathematical Formalism

The meta-learning objective over a batch of N tasks:

```
min_theta  (1/N) * sum_i  w_i * L(f_{theta'_i}, D_i^q)
```

where `theta'_i` is the result of k inner gradient steps:

```
theta_i^(0)   = theta
theta_i^(j+1) = theta_i^(j) - alpha * T(grad L(f_{theta_i^(j)}, D_i^s))
theta'_i      = theta_i^(k)
```

`T` is the composed gradient transform (identity by default; SAM when composed). `w_i` are per-task weights (uniform by default).

The algorithm variants differ only in how the outer gradient is computed:

- **MAML**: full backprop through inner steps (second-order)
- **FOMAML**: treats adapted params as constants (first-order)
- **Reptile**: `theta += beta * mean(theta'_i - theta)` (parameter interpolation, no query set)
