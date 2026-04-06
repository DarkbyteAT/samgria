"""Few-shot classification with random nonlinear decision boundaries.

Each task samples 2D points and labels them above/below a random
sinusoidal boundary: y > a*sin(bx + c) + d.  A meta-learner must
find an initialisation that adapts to new boundaries from 5 examples.

BCE loss has sharp curvature near the decision boundary, making this
a stress test for second-order meta-learning: MAML captures the
gradient rotation through the boundary, FOMAML misses it, and Reptile
has no query-loss signal at all.

Usage::

    uv run python examples/fewshot_classification.py
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch as T
import torch.nn as nn
import torch.optim as optim

from samgria import FOMAML, MAML, Reptile, meta_step, restore_state, save_state


# --- Config ---
DEVICE = T.device("cuda" if T.cuda.is_available() else "mps" if T.backends.mps.is_available() else "cpu")
OUTER_STEPS = 500
INNER_STEPS = 5
OUTER_LR = 3e-4
INNER_LR = 0.05
TASKS_PER_STEP = 5
EVAL_EVERY = 5
EVAL_TASKS = 20
SUPPORT_SIZE = 5
QUERY_SIZE = 15
RUN_DIR = Path("results/fewshot_classification") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")

sns.set_style("darkgrid")
sns.set_context("paper")
PALETTE = {"Baseline": "#9E9E9E", "FOMAML": "#2196F3", "MAML": "#FF5722", "Reptile": "#4CAF50"}


# --- Task generation ---
def random_boundary() -> tuple[float, float, float, float]:
    a = 0.5 + T.rand(1).item() * 2.0
    b = 0.5 + T.rand(1).item() * 3.0
    c = T.rand(1).item() * 2 * math.pi
    d = (T.rand(1).item() - 0.5) * 2.0
    return a, b, c, d


def generate_task(n: int, boundary: tuple[float, float, float, float]) -> tuple[T.Tensor, T.Tensor]:
    a, b, c, d = boundary
    x = (T.rand(n, 2, device=DEVICE) - 0.5) * 4.0  # points in [-2, 2]^2
    threshold = a * T.sin(b * x[:, 0] + c) + d
    labels = (x[:, 1] > threshold).float().unsqueeze(1)
    return x, labels


def sample_task() -> tuple[tuple[T.Tensor, T.Tensor], tuple[T.Tensor, T.Tensor]]:
    boundary = random_boundary()
    support = generate_task(SUPPORT_SIZE, boundary)
    query = generate_task(QUERY_SIZE, boundary)
    return support, query


# --- Model ---
def fresh_model() -> tuple[nn.Sequential, optim.Adam]:
    T.manual_seed(0)
    m = nn.Sequential(
        nn.Linear(2, 40), nn.ReLU(),
        nn.Linear(40, 40), nn.ReLU(),
        nn.Linear(40, 1),
    ).to(DEVICE)
    return m, optim.Adam(m.parameters(), lr=OUTER_LR)


def bce_loss(model: nn.Module):
    def fn(*batch: T.Tensor) -> T.Tensor:
        x, y = batch[0], batch[1]
        logits = model(x)
        return nn.functional.binary_cross_entropy_with_logits(logits, y)
    return fn


def accuracy(model: nn.Module, x: T.Tensor, y: T.Tensor) -> float:
    with T.no_grad():
        preds = (T.sigmoid(model(x)) > 0.5).float()
        return (preds == y).float().mean().item()


# --- Evaluation ---
def eval_adaptation(model, opt, algo) -> tuple[float, float]:
    """Adapt on EVAL_TASKS held-out tasks, return mean query loss and accuracy."""
    outer = save_state(model, opt)
    total_loss, total_acc = 0.0, 0.0

    for i in range(EVAL_TASKS):
        T.manual_seed(10000 + i)
        support, query = sample_task()
        adapted = algo.adapt(model, opt, bce_loss(model), support, inner_steps=INNER_STEPS)
        restore_state(model, opt, adapted.snapshot)
        with T.no_grad():
            logits = model(query[0])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, query[1]).item()
        total_acc += accuracy(model, query[0], query[1])
        total_loss += loss
        restore_state(model, opt, outer)

    return total_loss / EVAL_TASKS, total_acc / EVAL_TASKS


# --- State ---
loss_curves: dict[str, list[float]] = {}
acc_curves: dict[str, list[float]] = {}
eval_steps: list[int] = []


# --- Charts ---
def render() -> None:
    # Loss chart
    fig = plt.figure(figsize=(8, 5), dpi=150)
    for name, losses in loss_curves.items():
        style = "--" if name == "Baseline" else "-"
        sns.lineplot(x=eval_steps[:len(losses)], y=losses,
                     label=name, color=PALETTE[name], linewidth=2, linestyle=style)
    plt.xlabel("Outer Step")
    plt.ylabel("Query BCE Loss")
    plt.title("Few-Shot Classification: Query Loss", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "query_loss.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    # Accuracy chart
    fig = plt.figure(figsize=(8, 5), dpi=150)
    for name, accs in acc_curves.items():
        style = "--" if name == "Baseline" else "-"
        sns.lineplot(x=eval_steps[:len(accs)], y=accs,
                     label=name, color=PALETTE[name], linewidth=2, linestyle=style)
    plt.xlabel("Outer Step")
    plt.ylabel("Query Accuracy")
    plt.ylim(0.4, 1.0)
    plt.axhline(y=0.5, color="black", linestyle=":", alpha=0.3, label="Chance")
    plt.title("Few-Shot Classification: Query Accuracy", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "query_accuracy.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# --- Training ---
def train_baseline() -> None:
    model, opt = fresh_model()
    fomaml = FOMAML(inner_lr=INNER_LR)
    loss_curves["Baseline"] = []
    acc_curves["Baseline"] = []

    for step in range(OUTER_STEPS):
        support, _ = sample_task()
        opt.zero_grad()
        loss = bce_loss(model)(*support)
        loss.backward()
        opt.step()

        if (step + 1) % EVAL_EVERY == 0:
            q_loss, q_acc = eval_adaptation(model, opt, fomaml)
            loss_curves["Baseline"].append(q_loss)
            acc_curves["Baseline"].append(q_acc)
            if not eval_steps or eval_steps[-1] < step + 1:
                eval_steps.append(step + 1)
            render()
            if (step + 1) % 100 == 0:
                print(f"  Baseline step {step + 1:>4d}  acc: {q_acc:.3f}  loss: {q_loss:.4f}")


def train(name: str, algo) -> None:
    model, opt = fresh_model()
    loss_curves[name] = []
    acc_curves[name] = []

    for step in range(OUTER_STEPS):
        with meta_step(algo, model, opt, loss_fn=bce_loss(model), inner_steps=INNER_STEPS) as ms:
            for _ in range(TASKS_PER_STEP):
                support, query = sample_task()
                ms.task(support=support, query=query)

        if (step + 1) % EVAL_EVERY == 0:
            q_loss, q_acc = eval_adaptation(model, opt, algo)
            loss_curves[name].append(q_loss)
            acc_curves[name].append(q_acc)
            if not eval_steps or eval_steps[-1] < step + 1:
                eval_steps.append(step + 1)
            render()
            if (step + 1) % 100 == 0:
                print(f"  {name} step {step + 1:>4d}  acc: {q_acc:.3f}  loss: {q_loss:.4f}")


def train_reptile() -> None:
    model, opt = fresh_model()
    reptile = Reptile(inner_lr=INNER_LR, meta_lr=1.0)
    loss_curves["Reptile"] = []
    acc_curves["Reptile"] = []

    for step in range(OUTER_STEPS):
        with meta_step(reptile, model, opt, loss_fn=bce_loss(model), inner_steps=INNER_STEPS) as ms:
            for _ in range(TASKS_PER_STEP):
                support, _ = sample_task()
                ms.task(support=support)

        if (step + 1) % EVAL_EVERY == 0:
            q_loss, q_acc = eval_adaptation(model, opt, reptile)
            loss_curves["Reptile"].append(q_loss)
            acc_curves["Reptile"].append(q_acc)
            render()
            if (step + 1) % 100 == 0:
                print(f"  Reptile step {step + 1:>4d}  acc: {q_acc:.3f}  loss: {q_loss:.4f}")


# --- Main ---
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Few-Shot Classification with Nonlinear Boundaries")
print(f"Results: {RUN_DIR}")
print("=" * 60)
print(f"\n{SUPPORT_SIZE}-shot support, {QUERY_SIZE}-shot query, {OUTER_STEPS} outer steps")

print("\nBaseline...")
train_baseline()

print("\nFOMAML...")
train("FOMAML", FOMAML(inner_lr=INNER_LR))

print("\nMAML...")
train("MAML", MAML(inner_lr=INNER_LR))

print("\nReptile...")
train_reptile()

render()
print("\n" + "=" * 60)
print("Final query accuracy (higher = better):")
for name, accs in acc_curves.items():
    print(f"  {name:>8s}: {accs[-1]:.3f}")
print(f"\n  Charts: {RUN_DIR}")
