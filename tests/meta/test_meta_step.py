"""Behaviour-driven tests for the MetaStep builder and meta_step context manager.

These tests document the contract that MetaStep and meta_step must satisfy:
    - The context manager saves state on enter and steps on exit.
    - .task() adapts, computes query loss, and returns AdaptedState.
    - .step() with zero tasks raises ValueError.
    - Per-task overrides (inner_steps, grad_transforms, weight) work.
    - Task weighting scales query losses correctly.
    - Inner-loop regularisation adds a penalty to the inner loss.
    - Reptile works with query=None (no query set).
    - MAML's second-order gradients flow through the MetaStep API.
"""

from __future__ import annotations

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector

import pytest

from samgria.meta import FOMAML, MAML, Reptile
from samgria.meta.step import MetaStep, meta_step
from samgria.transforms.sam import SAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlp() -> nn.Sequential:
    T.manual_seed(42)
    return nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 1))


def _sinusoid_task(
    amplitude: float, phase: float, n: int = 10,
) -> tuple[T.Tensor, T.Tensor]:
    x = T.linspace(-5.0, 5.0, n).unsqueeze(1)
    y = amplitude * T.sin(x + phase)
    return x, y


def _mse_loss_fn(model: nn.Module):
    def loss_fn(*support: T.Tensor) -> T.Tensor:
        x, y = support[0], support[1]
        return ((model(x) - y) ** 2).mean()
    return loss_fn


# ---------------------------------------------------------------------------
# Context manager basics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_context_manager_applies_outer_update() -> None:
    """The context manager saves state on enter and applies outer update on exit."""
    # Given a model and FOMAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When we run a meta_step with one task
    x_s, y_s = _sinusoid_task(2.0, 1.0)
    x_q, y_q = _sinusoid_task(2.0, 1.0, n=5)
    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=3) as ms:
        ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # Then parameters have changed (outer update applied)
    params_after = parameters_to_vector(model.parameters()).detach()
    assert not T.equal(params_before, params_after)


@pytest.mark.unit
def test_step_with_zero_tasks_raises() -> None:
    """Calling .step() with no tasks raises ValueError."""
    # Given an empty MetaStep
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    ms = MetaStep(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=3)

    # Then step() raises
    with pytest.raises(ValueError, match="zero"):
        ms.step()


@pytest.mark.unit
def test_task_returns_adapted_state() -> None:
    """task() returns the AdaptedState for inspection."""
    # Given a MetaStep
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    x_s, y_s = _sinusoid_task(2.0, 1.0)
    x_q, y_q = _sinusoid_task(2.0, 1.0, n=5)
    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=3) as ms:
        result = ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # Then the result is an AdaptedState
    from samgria.state import AdaptedState
    assert isinstance(result, AdaptedState)


# ---------------------------------------------------------------------------
# Algorithm transparency — same consumer code, different algorithms
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fomaml_and_maml_produce_different_updates_via_meta_step() -> None:
    """MAML and FOMAML produce different outer updates through MetaStep."""
    # Given identical models
    T.manual_seed(42)
    model_maml = _make_mlp()
    opt_maml = optim.SGD(model_maml.parameters(), lr=0.01)

    T.manual_seed(42)
    model_fomaml = _make_mlp()
    opt_fomaml = optim.SGD(model_fomaml.parameters(), lr=0.01)

    x_s, y_s = _sinusoid_task(2.0, 1.0)
    x_q, y_q = _sinusoid_task(2.0, 1.0, n=5)

    # When MAML runs one outer step
    with meta_step(MAML(inner_lr=0.1), model_maml, opt_maml,
                   loss_fn=_mse_loss_fn(model_maml), inner_steps=5) as ms:
        ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # And FOMAML runs one outer step
    with meta_step(FOMAML(inner_lr=0.1), model_fomaml, opt_fomaml,
                   loss_fn=_mse_loss_fn(model_fomaml), inner_steps=5) as ms:
        ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # Then the resulting parameters differ
    p_maml = parameters_to_vector(model_maml.parameters()).detach()
    p_fomaml = parameters_to_vector(model_fomaml.parameters()).detach()
    assert not T.allclose(p_maml, p_fomaml, atol=1e-6)


@pytest.mark.unit
def test_reptile_works_with_no_query_set() -> None:
    """Reptile tasks have query=None — no query loss computed."""
    # Given a model and Reptile
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    reptile = Reptile(inner_lr=0.01, meta_lr=0.1)
    params_before = parameters_to_vector(model.parameters()).detach().clone()

    x_s, y_s = _sinusoid_task(2.0, 1.0)
    with meta_step(reptile, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=5) as ms:
        ms.task(support=(x_s, y_s))
        ms.task(support=_sinusoid_task(3.0, 0.5))

    # Then parameters changed
    params_after = parameters_to_vector(model.parameters()).detach()
    assert not T.equal(params_before, params_after)


# ---------------------------------------------------------------------------
# Per-task overrides
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_task_inner_steps_override() -> None:
    """Different inner_steps per task produces different adapted params."""
    # Given a model
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    x_s, y_s = _sinusoid_task(2.0, 1.0)
    x_q, y_q = _sinusoid_task(2.0, 1.0, n=5)

    # When we run tasks with different inner_steps
    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=1) as ms:
        result_1 = ms.task(support=(x_s, y_s), query=(x_q, y_q))
        result_10 = ms.task(support=(x_s, y_s), query=(x_q, y_q), inner_steps=10)

    # Then their adapted params differ
    assert not T.equal(result_1.snapshot.params, result_10.snapshot.params)


@pytest.mark.unit
def test_per_task_grad_transforms_override() -> None:
    """SAM on one task but not another produces different adapted params."""
    # Given a model
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    x_s, y_s = _sinusoid_task(2.0, 1.0)
    x_q, y_q = _sinusoid_task(2.0, 1.0, n=5)

    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=5) as ms:
        result_plain = ms.task(support=(x_s, y_s), query=(x_q, y_q))
        result_sam = ms.task(support=(x_s, y_s), query=(x_q, y_q),
                             grad_transforms=[SAM(rho=0.05)])

    assert not T.equal(result_plain.snapshot.params, result_sam.snapshot.params)


# ---------------------------------------------------------------------------
# Task weighting
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_task_weighting_affects_outer_update() -> None:
    """Differently weighted tasks produce a different outer update than uniform."""
    # Given identical starting models
    T.manual_seed(42)
    model_uniform = _make_mlp()
    opt_uniform = optim.SGD(model_uniform.parameters(), lr=0.01)

    T.manual_seed(42)
    model_weighted = _make_mlp()
    opt_weighted = optim.SGD(model_weighted.parameters(), lr=0.01)

    fomaml = FOMAML(inner_lr=0.01)
    task_a = (_sinusoid_task(1.0, 0.0), _sinusoid_task(1.0, 0.0, n=5))
    task_b = (_sinusoid_task(5.0, 1.0), _sinusoid_task(5.0, 1.0, n=5))

    # When uniform weights
    with meta_step(fomaml, model_uniform, opt_uniform,
                   loss_fn=_mse_loss_fn(model_uniform), inner_steps=3) as ms:
        ms.task(support=task_a[0], query=task_a[1])
        ms.task(support=task_b[0], query=task_b[1])

    # And heavily weighted toward task_b
    with meta_step(fomaml, model_weighted, opt_weighted,
                   loss_fn=_mse_loss_fn(model_weighted), inner_steps=3) as ms:
        ms.task(support=task_a[0], query=task_a[1], weight=0.1)
        ms.task(support=task_b[0], query=task_b[1], weight=10.0)

    # Then the resulting parameters differ
    p_uniform = parameters_to_vector(model_uniform.parameters()).detach()
    p_weighted = parameters_to_vector(model_weighted.parameters()).detach()
    assert not T.allclose(p_uniform, p_weighted, atol=1e-6)


# ---------------------------------------------------------------------------
# Inner-loop regularisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inner_reg_fn_affects_adapted_params() -> None:
    """Inner-loop regularisation changes the adapted params."""
    # Given a model
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    x_s, y_s = _sinusoid_task(2.0, 1.0)
    x_q, y_q = _sinusoid_task(2.0, 1.0, n=5)

    # When we adapt without regularisation
    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=5) as ms:
        result_plain = ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # And adapt with L2 regularisation toward base params
    def l2_reg(current_params: dict[str, T.Tensor], base_params: dict[str, T.Tensor]) -> T.Tensor:
        return T.stack([
            ((c - b) ** 2).sum()
            for c, b in zip(current_params.values(), base_params.values())
        ]).sum()

    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=5,
                   inner_reg_fn=l2_reg) as ms:
        result_reg = ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # Then the adapted params differ (regularisation pulled toward base)
    assert not T.equal(result_plain.snapshot.params, result_reg.snapshot.params)


@pytest.mark.unit
def test_inner_reg_fn_pulls_toward_base() -> None:
    """Strong L2 reg keeps adapted params closer to base than without reg."""
    # Given a model
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    base_params = parameters_to_vector(model.parameters()).detach().clone()
    x_s, y_s = _sinusoid_task(2.0, 1.0)
    x_q, y_q = _sinusoid_task(2.0, 1.0, n=5)

    # When we adapt without reg
    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=10) as ms:
        result_plain = ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # And with strong L2 reg
    def strong_l2(current: dict[str, T.Tensor], base: dict[str, T.Tensor]) -> T.Tensor:
        return 10.0 * T.stack([
            ((c - b) ** 2).sum() for c, b in zip(current.values(), base.values())
        ]).sum()

    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=10,
                   inner_reg_fn=strong_l2) as ms:
        result_reg = ms.task(support=(x_s, y_s), query=(x_q, y_q))

    # Then regularised params are closer to base
    dist_plain = (result_plain.snapshot.params - base_params).norm()
    dist_reg = (result_reg.snapshot.params - base_params).norm()
    assert dist_reg < dist_plain


# ---------------------------------------------------------------------------
# Multi-task integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_meta_step_multi_task_reduces_loss() -> None:
    """A full outer step with multiple tasks reduces held-out loss."""
    T.manual_seed(0)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    # Meta-train for 10 outer steps
    for _ in range(10):
        with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=3) as ms:
            for task_idx in range(5):
                T.manual_seed(task_idx * 100)
                amp = 0.5 + T.rand(1).item() * 4.5
                phase = T.rand(1).item() * 3.14
                support = _sinusoid_task(amp, phase)
                query = _sinusoid_task(amp, phase, n=5)
                ms.task(support=support, query=query)

    # Evaluate on held-out task
    x_test, y_test = _sinusoid_task(3.0, 1.0, n=20)
    with T.no_grad():
        loss_before_adapt = ((model(x_test) - y_test) ** 2).mean().item()

    # Adapt
    with meta_step(fomaml, model, optimizer, loss_fn=_mse_loss_fn(model), inner_steps=5) as ms:
        ms.task(support=_sinusoid_task(3.0, 1.0), query=_sinusoid_task(3.0, 1.0, n=5))

    # The outer step itself isn't adaptation — let's use adapt directly
    from samgria.state import restore_state
    result = fomaml.adapt(model, optimizer, _mse_loss_fn(model),
                          _sinusoid_task(3.0, 1.0), inner_steps=5)
    restore_state(model, optimizer, result.snapshot)

    with T.no_grad():
        loss_after_adapt = ((model(x_test) - y_test) ** 2).mean().item()

    assert loss_after_adapt < loss_before_adapt
