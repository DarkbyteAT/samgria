"""Effect of inner-loop regularisation on meta-learning stability.

Compares FOMAML and Reptile at different L2 regularisation strengths.
L2 reg penalises the adapted parameters for drifting from the base
initialisation: loss_inner = loss_task + lambda * ||theta' - theta||^2.

Usage::

    uv run python examples/inner_regularisation.py
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

from samgria import FOMAML, Reptile, meta_step, restore_state, save_state


# --- Config ---
DEVICE = T.device("cuda" if T.cuda.is_available() else "mps" if T.backends.mps.is_available() else "cpu")
OUTER_STEPS = 500
INNER_STEPS = 5
OUTER_LR = 3e-4
INNER_LR = 0.02
TASKS_PER_STEP = 5
EVAL_EVERY = 25
TEST_AMP, TEST_PHASE = 3.0, 1.0
LAMBDAS = [0.0, 0.01, 0.1, 0.5, 1.0]
RUN_DIR = Path("results/inner_regularisation") / datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")

sns.set_style("darkgrid")
sns.set_context("paper")


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


def l2_reg(lam: float):
    def reg(current: dict[str, T.Tensor], base: dict[str, T.Tensor]) -> T.Tensor:
        return lam * T.stack([((c - b) ** 2).sum() for c, b in zip(current.values(), base.values(), strict=True)]).sum()

    return reg


# --- State ---
fomaml_curves: dict[str, list[float]] = {}
reptile_curves: dict[str, list[float]] = {}
eval_steps: list[int] = []


# --- Charts ---
def render() -> None:
    fomaml_pal = sns.color_palette("Blues_d", len(fomaml_curves))
    reptile_pal = sns.color_palette("Greens_d", len(reptile_curves))

    for title, curves, palette, filename in [
        ("FOMAML: L2 Regularisation Sweep", fomaml_curves, fomaml_pal, "fomaml_reg_sweep.svg"),
        ("Reptile: L2 Regularisation Sweep", reptile_curves, reptile_pal, "reptile_reg_sweep.svg"),
    ]:
        fig = plt.figure(figsize=(8, 5), dpi=150)
        for i, (lam, losses) in enumerate(curves.items()):
            sns.lineplot(x=eval_steps[: len(losses)], y=losses, label=f"λ={lam}", color=palette[i], linewidth=2)
        plt.xlabel("Outer Step")
        plt.ylabel("Post-Adaptation MSE")
        plt.yscale("log")
        plt.title(title, fontweight="bold")
        plt.legend(title="λ")
        plt.tight_layout()
        plt.savefig(RUN_DIR / filename, format="svg", bbox_inches="tight")
        plt.close(fig)


# --- Training ---
def train_fomaml(label: str, reg_fn=None) -> None:
    model, opt = fresh_model()
    fomaml = FOMAML(inner_lr=INNER_LR)
    fomaml_curves[label] = []

    mse = eval_adaptation(model, opt, fomaml)
    fomaml_curves[label].append(mse)
    if not eval_steps:
        eval_steps.append(0)
    print(f"  FOMAML λ={label} step   0  loss: {mse:.4f}")

    for step in range(OUTER_STEPS):
        kwargs = {"loss_fn": loss_fn(model), "inner_steps": INNER_STEPS}
        if reg_fn is not None:
            kwargs["inner_reg_fn"] = reg_fn

        with meta_step(fomaml, model, opt, **kwargs) as ms:
            for _ in range(TASKS_PER_STEP):
                amp, phase = random_sinusoid()
                ms.task(support=sinusoid(amp, phase), query=sinusoid(amp, phase))

        if (step + 1) % EVAL_EVERY == 0:
            mse = eval_adaptation(model, opt, fomaml)
            fomaml_curves[label].append(mse)
            if not eval_steps or eval_steps[-1] < step + 1:
                eval_steps.append(step + 1)
            render()
            if (step + 1) % 100 == 0:
                print(f"  FOMAML λ={label} step {step + 1:>3d}  loss: {mse:.4f}")


def train_reptile(label: str, reg_fn=None) -> None:
    model, opt = fresh_model()
    reptile = Reptile(inner_lr=INNER_LR, meta_lr=1.0)
    reptile_curves[label] = []

    mse = eval_adaptation(model, opt, reptile)
    reptile_curves[label].append(mse)
    print(f"  Reptile λ={label} step   0  loss: {mse:.4f}")

    for step in range(OUTER_STEPS):
        kwargs = {"loss_fn": loss_fn(model), "inner_steps": INNER_STEPS}
        if reg_fn is not None:
            kwargs["inner_reg_fn"] = reg_fn

        with meta_step(reptile, model, opt, **kwargs) as ms:
            for _ in range(TASKS_PER_STEP):
                amp, phase = random_sinusoid()
                ms.task(support=sinusoid(amp, phase))

        if (step + 1) % EVAL_EVERY == 0:
            mse = eval_adaptation(model, opt, reptile)
            reptile_curves[label].append(mse)
            render()
            if (step + 1) % 100 == 0:
                print(f"  Reptile λ={label} step {step + 1:>3d}  loss: {mse:.4f}")


# --- Main ---
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Inner-Loop L2 Regularisation Sweep")
print(f"Device: {DEVICE}")
print(f"Results: {RUN_DIR}")
print(f"Lambdas: {LAMBDAS}")
print("=" * 60)

for lam in LAMBDAS:
    label = str(lam)
    reg = l2_reg(lam) if lam > 0 else None
    print(f"\nFOMAML λ={lam}...")
    train_fomaml(label, reg_fn=reg)

for lam in LAMBDAS:
    label = str(lam)
    reg = l2_reg(lam) if lam > 0 else None
    print(f"\nReptile λ={lam}...")
    train_reptile(label, reg_fn=reg)

render()
print("\n" + "=" * 60)
print("FOMAML final losses:")
for label, losses in fomaml_curves.items():
    print(f"  λ={label:>4s}: {losses[-1]:.4f}")
print("\nReptile final losses:")
for label, losses in reptile_curves.items():
    print(f"  λ={label:>4s}: {losses[-1]:.4f}")
print(f"\n  Charts: {RUN_DIR}")
