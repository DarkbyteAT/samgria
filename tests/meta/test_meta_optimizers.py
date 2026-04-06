"""Behaviour-driven tests for MetaOptimizer protocol and implementations.

These tests document the contract that MetaOptimizer, MAML, FOMAML, and
Reptile must satisfy.  Each test name describes a single observable
behaviour, structured as Given-When-Then.

Key guarantees under test:
    - MAML, FOMAML, and Reptile satisfy the MetaOptimizer protocol.
    - adapt() returns an AdaptedState without modifying outer state.
    - meta_step() computes and applies a meaningful outer-loop update.
    - MAML produces true second-order gradients via functional_call.
    - MAML and FOMAML produce different outer updates (second-order vs first).
    - FOMAML does NOT create second-order graphs (first-order approx).
    - Reptile uses parameter interpolation, not query-loss gradients.
    - GradientTransforms (e.g. SAM) compose inside the inner loop.
    - Inner loop defaults to sgd(lr) but is configurable via inner_step_fn.
    - Outer optimizer state is untouched after adapt().
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import pytest
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector

from samgria.meta import FOMAML, MAML, MetaOptimizer, Reptile
from samgria.state import AdaptedState, ParameterSnapshot, query_forward, restore_state, save_state
from samgria.transforms.sam import SAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlp() -> nn.Sequential:
    """Two-layer MLP with deterministic weights for meta-learning tests."""
    T.manual_seed(42)
    return nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 1))


def _sinusoid_task(amplitude: float, phase: float, n_samples: int = 10) -> tuple[T.Tensor, T.Tensor]:
    """Generate a sinusoid regression task: y = amplitude * sin(x + phase)."""
    x = T.linspace(-5.0, 5.0, n_samples).unsqueeze(1)
    y = amplitude * T.sin(x + phase)
    return x, y


def _mse_loss_fn(
    model: nn.Module,
) -> Callable[..., T.Tensor]:
    """Return a loss function closure over the model for use with adapt()."""

    def loss_fn(*support: T.Tensor) -> T.Tensor:
        x, y = support[0], support[1]
        pred = model(x)
        return ((pred - y) ** 2).mean()

    return loss_fn


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_maml_satisfies_meta_optimizer_protocol() -> None:
    """MAML is a runtime-checkable MetaOptimizer."""
    # Given a MAML instance
    maml = MAML(inner_lr=0.01)

    # Then it satisfies the MetaOptimizer protocol
    assert isinstance(maml, MetaOptimizer)


@pytest.mark.unit
def test_fomaml_satisfies_meta_optimizer_protocol() -> None:
    """FOMAML is a runtime-checkable MetaOptimizer."""
    # Given a FOMAML instance
    fomaml = FOMAML(inner_lr=0.01)

    # Then it satisfies the MetaOptimizer protocol
    assert isinstance(fomaml, MetaOptimizer)


@pytest.mark.unit
def test_reptile_satisfies_meta_optimizer_protocol() -> None:
    """Reptile is a runtime-checkable MetaOptimizer."""
    # Given a Reptile instance
    reptile = Reptile(inner_lr=0.01, meta_lr=0.1)

    # Then it satisfies the MetaOptimizer protocol
    assert isinstance(reptile, MetaOptimizer)


# ---------------------------------------------------------------------------
# AdaptedState structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adapt_returns_adapted_state() -> None:
    """adapt() returns an AdaptedState containing a ParameterSnapshot."""
    # Given a model with FOMAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    # When adapt() runs
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    result = fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)

    # Then the result is an AdaptedState with a snapshot
    assert isinstance(result, AdaptedState)
    assert isinstance(result.snapshot, ParameterSnapshot)


@pytest.mark.unit
def test_fomaml_adapted_state_has_no_live_params() -> None:
    """FOMAML's AdaptedState has live_params=None (first-order, no graph)."""
    # Given a model with FOMAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    # When adapt() runs
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    result = fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)

    # Then live_params is None
    assert result.live_params is None


@pytest.mark.unit
def test_maml_adapted_state_has_live_params() -> None:
    """MAML's AdaptedState has graph-connected live_params for second-order."""
    # Given a model with MAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    maml = MAML(inner_lr=0.01)

    # When adapt() runs
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    result = maml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)

    # Then live_params is populated with graph-connected tensors
    assert result.live_params is not None
    assert isinstance(result.live_params, dict)
    # And the tensors have grad_fn (they're part of the computation graph)
    for name, p in result.live_params.items():
        assert p.grad_fn is not None, f"live_params['{name}'] has no grad_fn — graph was severed"


# ---------------------------------------------------------------------------
# State isolation — adapt() must not modify outer state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adapt_does_not_modify_outer_parameters() -> None:
    """adapt() returns an adapted state without changing the caller's model."""
    # Given a model, optimizer, and FOMAML meta-optimizer
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    outer_params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When adapt() runs 5 inner steps
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=5)

    # Then the outer parameters are unchanged
    outer_params_after = parameters_to_vector(model.parameters()).detach()
    assert T.equal(outer_params_before, outer_params_after)


@pytest.mark.unit
def test_maml_adapt_does_not_modify_outer_parameters() -> None:
    """MAML's functional inner loop also preserves outer state."""
    # Given a model and MAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    maml = MAML(inner_lr=0.01)
    outer_params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When adapt() runs
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    maml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=5)

    # Then the outer parameters are unchanged
    outer_params_after = parameters_to_vector(model.parameters()).detach()
    assert T.equal(outer_params_before, outer_params_after)


@pytest.mark.unit
def test_adapt_does_not_modify_outer_optimizer_state() -> None:
    """adapt() preserves the outer optimizer's momentum/variance buffers."""
    # Given a model trained for a few steps (so Adam has state)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x_train, y_train = _sinusoid_task(amplitude=1.0, phase=0.0)
    for _ in range(3):
        optimizer.zero_grad()
        loss = ((model(x_train) - y_train) ** 2).mean()
        loss.backward()
        optimizer.step()

    optim_state_before = copy.deepcopy(optimizer.state_dict())

    # When adapt() runs
    fomaml = FOMAML(inner_lr=0.01)
    fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x_train, y_train), inner_steps=5)

    # Then the optimizer state is unchanged
    optim_state_after = optimizer.state_dict()
    for key in optim_state_before["state"]:
        for buf_name in optim_state_before["state"][key]:
            before = optim_state_before["state"][key][buf_name]
            after = optim_state_after["state"][key][buf_name]
            if isinstance(before, T.Tensor):
                assert T.equal(before, after), f"Optimizer state changed: param {key}, buffer {buf_name}"
            else:
                assert before == after


@pytest.mark.unit
def test_adapt_clears_gradients_after_returning() -> None:
    """adapt() leaves no stale gradients on the model's parameters."""
    # Given a model with FOMAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    # When adapt() runs
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)

    # Then all gradients are None
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.unit
def test_adapted_snapshot_differs_from_outer_state() -> None:
    """The adapted snapshot holds different parameters than the outer state."""
    # Given a model
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    outer_params = parameters_to_vector(model.parameters()).detach().clone()

    # When adapt() runs
    fomaml = FOMAML(inner_lr=0.01)
    x, y = _sinusoid_task(amplitude=2.0, phase=1.0)
    result = fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=5)

    # Then the adapted params differ from outer params
    assert not T.equal(result.snapshot.params, outer_params)


# ---------------------------------------------------------------------------
# Configurable inner optimizer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adapt_with_mutation_optimizer() -> None:
    """adapt() accepts mutation_optimizer() to use standard PyTorch optimizers."""
    # Given a model and FOMAML
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    x, y = _sinusoid_task(amplitude=1.5, phase=0.5)
    outer_params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When adapt() runs with a mutation_optimizer wrapping Adam
    from samgria.meta.protocol import mutation_optimizer

    result = fomaml.adapt(
        model,
        optimizer,
        _mse_loss_fn(model),
        (x, y),
        inner_steps=5,
        inner_step_fn=mutation_optimizer(lambda p: optim.Adam(p, lr=0.01)),
    )

    # Then the adapted params differ from outer
    assert not T.equal(result.snapshot.params, outer_params_before)
    # And outer state is still isolated
    outer_params_after = parameters_to_vector(model.parameters()).detach()
    assert T.equal(outer_params_before, outer_params_after)


@pytest.mark.unit
def test_custom_inner_step_produces_different_trajectory() -> None:
    """A custom inner_step_fn produces different adapted params than default SGD."""
    # Given a deterministic model and task
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    x, y = _sinusoid_task(amplitude=1.5, phase=0.5)

    # When we adapt with default SGD
    snap_sgd = fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=5)

    # And adapt with mutation_optimizer wrapping Adam
    from samgria.meta.protocol import mutation_optimizer

    snap_adam = fomaml.adapt(
        model,
        optimizer,
        _mse_loss_fn(model),
        (x, y),
        inner_steps=5,
        inner_step_fn=mutation_optimizer(lambda p: optim.Adam(p, lr=0.01)),
    )

    # Then the trajectories differ
    assert not T.equal(snap_sgd.snapshot.params, snap_adam.snapshot.params)


@pytest.mark.unit
def test_maml_with_custom_inner_step_fn() -> None:
    """MAML accepts a differentiable inner_step_fn and preserves the graph."""
    # Given a model and MAML with a custom differentiable SGD step
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    maml = MAML(inner_lr=0.01)
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)

    # A differentiable step function (graph flows through)
    from samgria.meta.protocol import sgd

    result = maml.adapt(
        model,
        optimizer,
        _mse_loss_fn(model),
        (x, y),
        inner_steps=3,
        inner_step_fn=sgd(lr=0.05),  # different LR than default
    )

    # Then live_params are present (graph preserved)
    assert isinstance(result, AdaptedState)
    assert result.live_params is not None


# ---------------------------------------------------------------------------
# MAML — second-order gradients
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_maml_inner_loop_creates_computation_graph() -> None:
    """MAML uses create_graph=True so gradients flow through inner steps."""
    # Given a MAML instance
    maml = MAML(inner_lr=0.01)

    # Then it uses second-order gradients
    assert maml.create_graph is True


@pytest.mark.unit
def test_maml_query_forward_produces_differentiable_loss() -> None:
    """query_forward with MAML's live_params produces a graph-connected loss.

    The loss tensor should have a grad_fn — proving the computation graph
    flows through the inner-loop steps into the query loss.
    """
    # Given a model and MAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    maml = MAML(inner_lr=0.01)

    # When we adapt and compute a query loss via query_forward
    x_s, y_s = _sinusoid_task(amplitude=1.0, phase=0.0)
    result = maml.adapt(model, optimizer, _mse_loss_fn(model), (x_s, y_s), inner_steps=3)

    x_q, y_q = _sinusoid_task(amplitude=1.0, phase=0.0, n_samples=5)
    pred = query_forward(model, result, x_q)
    query_loss = ((pred - y_q) ** 2).mean()

    # Then the query loss has a grad_fn (graph is alive)
    assert query_loss.grad_fn is not None


@pytest.mark.unit
def test_maml_and_fomaml_produce_different_outer_gradients() -> None:
    """MAML's second-order gradients differ from FOMAML's first-order.

    This is the critical test: with identical starting parameters and
    tasks, MAML and FOMAML should produce different outer-loop gradients
    because MAML's backward pass includes Hessian-vector products that
    FOMAML drops.
    """
    # Given identical models for MAML and FOMAML
    T.manual_seed(42)
    model_maml = _make_mlp()
    opt_maml = optim.SGD(model_maml.parameters(), lr=0.01)

    T.manual_seed(42)
    model_fomaml = _make_mlp()
    opt_fomaml = optim.SGD(model_fomaml.parameters(), lr=0.01)

    maml = MAML(inner_lr=0.1)
    fomaml = FOMAML(inner_lr=0.1)

    x_s, y_s = _sinusoid_task(amplitude=2.0, phase=1.0)
    x_q, y_q = _sinusoid_task(amplitude=2.0, phase=1.0, n_samples=5)

    # When MAML adapts and computes query loss with graph
    save_state(model_maml, opt_maml)
    result_maml = maml.adapt(
        model_maml,
        opt_maml,
        _mse_loss_fn(model_maml),
        (x_s, y_s),
        inner_steps=5,
    )
    pred_maml = query_forward(model_maml, result_maml, x_q)
    ql_maml = ((pred_maml - y_q) ** 2).mean()

    opt_maml.zero_grad()
    ql_maml.backward()
    maml_grads = parameters_to_vector([p.grad for p in model_maml.parameters() if p.grad is not None]).clone()

    # And FOMAML adapts and computes query loss without graph
    save_state(model_fomaml, opt_fomaml)
    result_fomaml = fomaml.adapt(
        model_fomaml,
        opt_fomaml,
        _mse_loss_fn(model_fomaml),
        (x_s, y_s),
        inner_steps=5,
    )
    # For FOMAML, restore adapted params to compute query loss
    restore_state(model_fomaml, opt_fomaml, result_fomaml.snapshot)
    ql_fomaml = ((model_fomaml(x_q) - y_q) ** 2).mean()

    opt_fomaml.zero_grad()
    ql_fomaml.backward()
    fomaml_grads = parameters_to_vector([p.grad for p in model_fomaml.parameters() if p.grad is not None]).clone()

    # Then the outer gradients differ (second-order terms in MAML)
    assert not T.allclose(maml_grads, fomaml_grads, atol=1e-6), (
        "MAML and FOMAML produced identical outer gradients — second-order graph is not flowing through"
    )


# ---------------------------------------------------------------------------
# FOMAML — first-order approximation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fomaml_does_not_create_higher_order_graph() -> None:
    """FOMAML uses create_graph=False — no second-order gradient computation."""
    # Given a FOMAML instance
    fomaml = FOMAML(inner_lr=0.01)

    # Then it does NOT use second-order gradients
    assert fomaml.create_graph is False


@pytest.mark.unit
def test_fomaml_adapted_params_are_leaf_tensors() -> None:
    """FOMAML's adapted snapshot contains detached leaf tensors."""
    # Given a model with FOMAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    # When adapt() runs
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    result = fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)

    # Then the params tensor is a leaf (no grad_fn)
    assert result.snapshot.params.grad_fn is None
    assert not result.snapshot.params.requires_grad


# ---------------------------------------------------------------------------
# Reptile — parameter interpolation outer update
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reptile_meta_step_does_not_require_query_losses() -> None:
    """Reptile's meta_step() works without query losses."""
    # Given a model, outer optimizer, and Reptile
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    reptile = Reptile(inner_lr=0.01, meta_lr=0.1)

    # When we adapt on two tasks
    base_snapshot = save_state(model, optimizer)
    adapted = []
    for amp in [1.0, 2.0]:
        x, y = _sinusoid_task(amplitude=amp, phase=0.0)
        result = reptile.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)
        adapted.append(result)

    # Then meta_step succeeds with query_losses=None (the default)
    reptile.meta_step(model, optimizer, base_snapshot, adapted)

    # And parameters have changed (outer update applied)
    new_params = parameters_to_vector(model.parameters()).detach()
    assert not T.equal(new_params, base_snapshot.params)


@pytest.mark.unit
def test_reptile_outer_update_interpolates_toward_adapted_params() -> None:
    """Reptile's outer update moves parameters toward the mean of adapted params."""
    # Given a model and Reptile
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    reptile = Reptile(inner_lr=0.01, meta_lr=0.5)

    # When we adapt on two tasks
    base_snapshot = save_state(model, optimizer)
    outer_params = base_snapshot.params.clone()
    adapted = []
    for amp, phase in [(1.0, 0.0), (2.0, math.pi / 4)]:
        x, y = _sinusoid_task(amplitude=amp, phase=phase)
        result = reptile.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=5)
        adapted.append(result)

    # And apply Reptile outer step
    reptile.meta_step(model, optimizer, base_snapshot, adapted)
    new_params = parameters_to_vector(model.parameters()).detach()

    # Then the update direction correlates with mean(adapted - outer)
    mean_diff = T.stack([a.snapshot.params - outer_params for a in adapted]).mean(dim=0)
    actual_update = new_params - outer_params

    cosine_sim = T.nn.functional.cosine_similarity(actual_update.unsqueeze(0), mean_diff.unsqueeze(0))
    assert cosine_sim.item() > 0.9, f"Reptile update direction diverges: cosine_sim={cosine_sim.item():.4f}"


# ---------------------------------------------------------------------------
# GradientTransform composability (SAM in inner loop)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fomaml_with_sam_produces_different_adapted_params() -> None:
    """FOMAML + SAM in the inner loop produces different adapted params than without."""
    # Given a deterministic model and task
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x, y = _sinusoid_task(amplitude=1.5, phase=0.5)

    # When we adapt without SAM
    fomaml = FOMAML(inner_lr=0.01)
    snap_plain = fomaml.adapt(
        model,
        optimizer,
        _mse_loss_fn(model),
        (x, y),
        inner_steps=5,
    )

    # And adapt with SAM (same starting point due to state isolation)
    snap_sam = fomaml.adapt(
        model,
        optimizer,
        _mse_loss_fn(model),
        (x, y),
        inner_steps=5,
        grad_transforms=[SAM(rho=0.05)],
    )

    # Then the adapted params differ
    assert not T.equal(snap_plain.snapshot.params, snap_sam.snapshot.params)


@pytest.mark.unit
def test_maml_with_sam_runs_without_error() -> None:
    """MAML + SAM in the inner loop runs without error."""
    # Given a model and MAML
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    maml = MAML(inner_lr=0.01)
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)

    # When adapt() runs with SAM — it should not raise
    result = maml.adapt(
        model,
        optimizer,
        _mse_loss_fn(model),
        (x, y),
        inner_steps=3,
        grad_transforms=[SAM(rho=0.05)],
    )

    # Then we get a valid AdaptedState with live params
    assert isinstance(result, AdaptedState)
    assert result.live_params is not None


# ---------------------------------------------------------------------------
# query_forward utility
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_query_forward_with_live_params_uses_functional_call() -> None:
    """query_forward routes through functional_call when live_params is set."""
    # Given a MAML adapted state with live_params (using large inner_lr
    # and many steps so adapted params diverge visibly from outer)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    maml = MAML(inner_lr=0.1)
    x, y = _sinusoid_task(amplitude=3.0, phase=1.0)
    result = maml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=10)

    # When query_forward is called on the sinusoid's input range
    x_q = T.linspace(-5.0, 5.0, 20).unsqueeze(1)
    pred = query_forward(model, result, x_q)

    # Then the output differs from the outer model (adapted params used)
    with T.no_grad():
        outer_pred = model(x_q)
    assert not T.allclose(pred.detach(), outer_pred, atol=1e-4), (
        "query_forward produced same output as outer model — functional_call is not routing through adapted params"
    )


@pytest.mark.unit
def test_query_forward_without_live_params_uses_model_directly() -> None:
    """query_forward falls back to model() when live_params is None."""
    # Given a FOMAML adapted state (no live_params)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)
    x, y = _sinusoid_task(amplitude=1.0, phase=0.0)
    result = fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)

    # When query_forward is called (model still at outer params)
    x_q = T.randn(5, 1)
    pred = query_forward(model, result, x_q)

    # Then the output matches a direct model call
    with T.no_grad():
        direct_pred = model(x_q)
    assert T.equal(pred.detach(), direct_pred)


# ---------------------------------------------------------------------------
# Integration: sinusoid regression smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fomaml_reduces_loss_on_held_out_sinusoid() -> None:
    """FOMAML adapts to a new sinusoid and reduces held-out loss."""
    T.manual_seed(0)
    model = _make_mlp()
    outer_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    # Meta-train on 5 random sinusoid tasks for 10 outer steps
    for _ in range(10):
        base_snapshot = save_state(model, outer_optimizer)
        adapted = []
        query_losses = []

        for task_idx in range(5):
            T.manual_seed(task_idx * 100)
            amp = 0.5 + T.rand(1).item() * 4.5
            phase = T.rand(1).item() * math.pi

            x_support, y_support = _sinusoid_task(amp, phase, n_samples=10)
            result = fomaml.adapt(
                model,
                outer_optimizer,
                _mse_loss_fn(model),
                (x_support, y_support),
                inner_steps=1,
            )
            adapted.append(result)

            # Evaluate adapted model on query set
            x_query, y_query = _sinusoid_task(amp, phase, n_samples=10)
            restore_state(model, outer_optimizer, result.snapshot)
            query_loss = ((model(x_query) - y_query) ** 2).mean()
            query_losses.append(query_loss)
            restore_state(model, outer_optimizer, base_snapshot)

        fomaml.meta_step(
            model,
            outer_optimizer,
            base_snapshot,
            adapted,
            query_losses=query_losses,
        )

    # Test on held-out sinusoid
    test_amp, test_phase = 3.0, 1.0
    x_test, y_test = _sinusoid_task(test_amp, test_phase, n_samples=20)

    with T.no_grad():
        loss_before = ((model(x_test) - y_test) ** 2).mean().item()

    x_support, y_support = _sinusoid_task(test_amp, test_phase, n_samples=10)
    result = fomaml.adapt(
        model,
        outer_optimizer,
        _mse_loss_fn(model),
        (x_support, y_support),
        inner_steps=5,
    )
    restore_state(model, outer_optimizer, result.snapshot)
    with T.no_grad():
        loss_after = ((model(x_test) - y_test) ** 2).mean().item()

    assert loss_after < loss_before, f"FOMAML failed to reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"


@pytest.mark.integration
def test_reptile_reduces_loss_on_held_out_sinusoid() -> None:
    """Reptile adapts to a new sinusoid and reduces held-out loss."""
    T.manual_seed(0)
    model = _make_mlp()
    outer_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    reptile = Reptile(inner_lr=0.01, meta_lr=1.0)

    for _ in range(20):
        base_snapshot = save_state(model, outer_optimizer)
        adapted = []

        for task_idx in range(5):
            T.manual_seed(task_idx * 100)
            amp = 0.5 + T.rand(1).item() * 4.5
            phase = T.rand(1).item() * math.pi

            x_support, y_support = _sinusoid_task(amp, phase, n_samples=10)
            result = reptile.adapt(
                model,
                outer_optimizer,
                _mse_loss_fn(model),
                (x_support, y_support),
                inner_steps=5,
            )
            adapted.append(result)

        reptile.meta_step(model, outer_optimizer, base_snapshot, adapted)

    # Test on held-out sinusoid
    test_amp, test_phase = 3.0, 1.0
    x_test, y_test = _sinusoid_task(test_amp, test_phase, n_samples=20)

    with T.no_grad():
        loss_before = ((model(x_test) - y_test) ** 2).mean().item()

    x_support, y_support = _sinusoid_task(test_amp, test_phase, n_samples=10)
    result = reptile.adapt(
        model,
        outer_optimizer,
        _mse_loss_fn(model),
        (x_support, y_support),
        inner_steps=10,
    )
    restore_state(model, outer_optimizer, result.snapshot)
    with T.no_grad():
        loss_after = ((model(x_test) - y_test) ** 2).mean().item()

    assert loss_after < loss_before, f"Reptile failed to reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"


@pytest.mark.integration
def test_multi_task_meta_step_produces_parameter_update() -> None:
    """A full meta_step with multiple tasks produces a non-zero parameter update."""
    # Given a model and FOMAML
    T.manual_seed(42)
    model = _make_mlp()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fomaml = FOMAML(inner_lr=0.01)

    # When we run a full outer step with 3 tasks
    base_snapshot = save_state(model, optimizer)
    adapted = []
    query_losses = []

    for amp in [1.0, 2.0, 3.0]:
        x, y = _sinusoid_task(amplitude=amp, phase=0.0)
        result = fomaml.adapt(model, optimizer, _mse_loss_fn(model), (x, y), inner_steps=3)
        adapted.append(result)

        # Compute query loss at adapted point
        restore_state(model, optimizer, result.snapshot)
        x_q, y_q = _sinusoid_task(amplitude=amp, phase=0.0, n_samples=5)
        q_loss = ((model(x_q) - y_q) ** 2).mean()
        query_losses.append(q_loss)
        restore_state(model, optimizer, base_snapshot)

    params_before_step = parameters_to_vector(model.parameters()).detach().clone()
    fomaml.meta_step(
        model,
        optimizer,
        base_snapshot,
        adapted,
        query_losses=query_losses,
    )
    params_after_step = parameters_to_vector(model.parameters()).detach()

    # Then parameters changed (outer update was applied)
    assert not T.equal(params_before_step, params_after_step)
