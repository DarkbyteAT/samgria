"""Train FOMAML, MAML, and Reptile on sinusoid regression.

Writes SVG charts to a timestamped results folder after every eval step.
Open the SVGs in a browser to watch curves build in real time.

Usage::

    uv run python examples/meta_learning_sinusoid.py
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
OUTER_STEPS = 1000
INNER_STEPS = 5
OUTER_LR = 3e-4
INNER_LR = 0.01
TASKS_PER_STEP = 5
EVAL_EVERY = 5
TEST_AMP, TEST_PHASE = 3.0, 1.0
RUN_DIR = Path("results/meta_sinusoid") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")

sns.set_style("darkgrid")
sns.set_context("paper")
PALETTE = {"Baseline": "#9E9E9E", "FOMAML": "#2196F3", "MAML": "#FF5722", "Reptile": "#4CAF50"}


# --- Primitives ---
def sinusoid(amp: float, phase: float, n: int = 10) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(-5.0, 5.0, n, device=DEVICE).unsqueeze(1)
    return x, amp * T.sin(x + phase)


def random_sinusoid() -> tuple[float, float]:
    return 0.5 + T.rand(1).item() * 4.5, T.rand(1).item() * math.pi


def fresh_model() -> tuple[nn.Sequential, optim.Adam]:
    T.manual_seed(0)
    m = nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, 1)).to(DEVICE)
    return m, optim.Adam(m.parameters(), lr=OUTER_LR)


def loss_fn(model: nn.Module):
    def fn(*batch: T.Tensor) -> T.Tensor:
        return ((model(batch[0]) - batch[1]) ** 2).mean()

    return fn


def eval_adaptation(model, opt, algo) -> float:
    outer = save_state(model, opt)
    adapted = algo.adapt(model, opt, loss_fn(model), sinusoid(TEST_AMP, TEST_PHASE), inner_steps=10)
    restore_state(model, opt, adapted.snapshot)
    with T.no_grad():
        x_t, y_t = sinusoid(TEST_AMP, TEST_PHASE, n=50)
        mse = ((model(x_t) - y_t) ** 2).mean().item()
    restore_state(model, opt, outer)
    return mse


# --- State ---
curves: dict[str, list[float]] = {}
eval_steps: list[int] = []


# --- Charts ---
def render_loss() -> None:
    fig = plt.figure(figsize=(8, 5), dpi=150)
    for name, losses in curves.items():
        style = "--" if name == "Baseline" else "-"
        sns.lineplot(
            x=eval_steps[: len(losses)], y=losses, label=name, color=PALETTE[name], linewidth=2, linestyle=style
        )
    plt.xlabel("Outer Step")
    plt.ylabel("Post-Adaptation MSE")
    plt.yscale("log")
    plt.title("Meta-Learning on Sinusoid Regression", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "adaptation_loss.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def render_gap() -> None:
    fomaml, maml = curves.get("FOMAML", []), curves.get("MAML", [])
    n = min(len(fomaml), len(maml))
    if n < 2:
        return
    gap = [abs(maml[i] - fomaml[i]) for i in range(n)]
    fig = plt.figure(figsize=(8, 5), dpi=150)
    sns.lineplot(x=eval_steps[:n], y=gap, color="#9C27B0", linewidth=2)
    plt.fill_between(eval_steps[:n], gap, alpha=0.15, color="#9C27B0")
    plt.xlabel("Outer Step")
    plt.ylabel("|MAML − FOMAML|")
    plt.title("Second-Order Gap Over Training", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "maml_fomaml_gap.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def render() -> None:
    render_loss()
    render_gap()


# --- Baseline ---
def train_baseline() -> None:
    """Train on random sinusoids WITHOUT meta-learning, just regular SGD."""
    model, opt = fresh_model()
    fomaml = FOMAML(inner_lr=INNER_LR)
    curves["Baseline"] = []

    mse = eval_adaptation(model, opt, fomaml)
    curves["Baseline"].append(mse)
    eval_steps.append(0)
    print(f"  Baseline step   0  loss: {mse:.4f}")

    for step in range(OUTER_STEPS):
        # Regular training: just do gradient descent on a random sinusoid
        amp, phase = random_sinusoid()
        x, y = sinusoid(amp, phase)
        opt.zero_grad()
        loss = loss_fn(model)(x, y)
        loss.backward()
        opt.step()

        if (step + 1) % EVAL_EVERY == 0:
            mse = eval_adaptation(model, opt, fomaml)
            curves["Baseline"].append(mse)
            if not eval_steps or eval_steps[-1] < step + 1:
                eval_steps.append(step + 1)
            render()
            if (step + 1) % 50 == 0:
                print(f"  Baseline step {step + 1:>3d}  loss: {mse:.4f}")


# --- Training ---
def train(name: str, algo) -> None:
    model, opt = fresh_model()
    curves[name] = []

    mse = eval_adaptation(model, opt, algo)
    curves[name].append(mse)
    print(f"  {name} step   0  loss: {mse:.4f}")

    for step in range(OUTER_STEPS):
        with meta_step(algo, model, opt, loss_fn=loss_fn(model), inner_steps=INNER_STEPS) as ms:
            for _ in range(TASKS_PER_STEP):
                amp, phase = random_sinusoid()
                ms.task(support=sinusoid(amp, phase), query=sinusoid(amp, phase))

        if (step + 1) % EVAL_EVERY == 0:
            mse = eval_adaptation(model, opt, algo)
            curves[name].append(mse)
            if not eval_steps or eval_steps[-1] < step + 1:
                eval_steps.append(step + 1)
            render()
            if (step + 1) % 50 == 0:
                print(f"  {name} step {step + 1:>3d}  loss: {mse:.4f}")


def train_reptile() -> None:
    model, opt = fresh_model()
    reptile = Reptile(inner_lr=INNER_LR, meta_lr=1.0)
    curves["Reptile"] = []

    for step in range(OUTER_STEPS):
        with meta_step(reptile, model, opt, loss_fn=loss_fn(model), inner_steps=INNER_STEPS) as ms:
            for _ in range(TASKS_PER_STEP):
                amp, phase = random_sinusoid()
                ms.task(support=sinusoid(amp, phase))

        if (step + 1) % EVAL_EVERY == 0:
            mse = eval_adaptation(model, opt, reptile)
            curves["Reptile"].append(mse)
            render()
            if (step + 1) % 50 == 0:
                print(f"  Reptile step {step + 1:>3d}  loss: {mse:.4f}")


# --- Main ---
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Meta-Learning on Sinusoid Regression")
print(f"Results: {RUN_DIR}")
print("=" * 60)

print("\nBaseline (regular SGD)...")
train_baseline()

print("FOMAML...")
train("FOMAML", FOMAML(inner_lr=INNER_LR))

print("\nMAML...")
train("MAML", MAML(inner_lr=INNER_LR))

print("\nReptile...")
train_reptile()

render()
print("\n" + "=" * 60)
for name, losses in curves.items():
    print(f"  {name:>8s}: {losses[-1]:.4f}")
print(f"\n  MAML-FOMAML gap: {abs(curves['MAML'][-1] - curves['FOMAML'][-1]):.6f}")
print(f"  Charts: {RUN_DIR}")
