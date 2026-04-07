"""Tests for SAM, ASAM, LAMPRollback transforms and composable pipeline."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as T
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from samgria import ASAM, SAM, GradientTransform, LAMPRollback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> nn.Linear:
    T.manual_seed(0)
    return nn.Linear(4, 2)


def _make_loss_fn(model: nn.Module) -> tuple[Callable[..., T.Tensor], tuple[T.Tensor, T.Tensor]]:
    """Return a closure over *model* and a fixed (x, y) batch."""
    T.manual_seed(1)
    x = T.randn(8, 4)
    y = T.randn(8, 2)

    def loss_fn(x: T.Tensor, y: T.Tensor) -> T.Tensor:
        return ((model(x) - y) ** 2).mean()

    return loss_fn, (x, y)


def _vanilla_grad(model: nn.Module, loss_fn: object, batch: tuple[T.Tensor, T.Tensor]) -> T.Tensor:
    """Compute a plain backward pass and return the flattened gradient."""
    model.zero_grad()
    loss = loss_fn(*batch)  # type: ignore[operator]
    loss.backward()
    return T.cat([p.grad.view(-1).clone() for p in model.parameters()])


# ---------------------------------------------------------------------------
# SAM tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sam_conforms_to_protocol() -> None:
    # Given a SAM instance
    sam = SAM()

    # Then it satisfies the GradientTransform protocol
    assert isinstance(sam, GradientTransform)


@pytest.mark.unit
def test_sam_parameters_restored_after_apply() -> None:
    # Given a model with an initial gradient
    model = _make_model()
    loss_fn, batch = _make_loss_fn(model)
    loss = loss_fn(*batch)
    loss.backward()
    params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When SAM.apply is called
    SAM(rho=0.05).apply(model, loss_fn, batch)

    # Then model parameters are restored to their original values
    params_after = parameters_to_vector(model.parameters()).detach()
    assert T.allclose(params_before, params_after, atol=1e-6)


@pytest.mark.unit
def test_sam_gradient_differs_from_vanilla() -> None:
    # Given a model with vanilla gradient
    model = _make_model()
    loss_fn, batch = _make_loss_fn(model)
    vanilla = _vanilla_grad(model, loss_fn, batch)

    # When SAM.apply overwrites the gradient
    model.zero_grad()
    loss = loss_fn(*batch)
    loss.backward()
    SAM(rho=0.05).apply(model, loss_fn, batch)
    sam_grad = T.cat([p.grad.view(-1).clone() for p in model.parameters()])

    # Then the SAM gradient differs from the vanilla gradient
    assert not T.allclose(vanilla, sam_grad, atol=1e-7)


@pytest.mark.unit
def test_sam_gradient_shape_preserved() -> None:
    # Given a model
    model = _make_model()
    loss_fn, batch = _make_loss_fn(model)
    loss = loss_fn(*batch)
    loss.backward()

    # When SAM.apply is called
    SAM(rho=0.05).apply(model, loss_fn, batch)

    # Then each parameter's .grad has the same shape as the parameter
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape


# ---------------------------------------------------------------------------
# ASAM tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_asam_conforms_to_protocol() -> None:
    # Given an ASAM instance
    asam = ASAM()

    # Then it satisfies the GradientTransform protocol
    assert isinstance(asam, GradientTransform)


@pytest.mark.unit
def test_asam_parameters_restored_after_apply() -> None:
    # Given a model with an initial gradient
    model = _make_model()
    loss_fn, batch = _make_loss_fn(model)
    loss = loss_fn(*batch)
    loss.backward()
    params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When ASAM.apply is called
    ASAM(rho=0.05).apply(model, loss_fn, batch)

    # Then model parameters are restored to their original values
    params_after = parameters_to_vector(model.parameters()).detach()
    assert T.allclose(params_before, params_after, atol=1e-6)


@pytest.mark.unit
def test_asam_gradient_differs_from_sam() -> None:
    # Given identical models with initial gradients
    model_sam = _make_model()
    loss_fn_sam, batch_sam = _make_loss_fn(model_sam)
    loss = loss_fn_sam(*batch_sam)
    loss.backward()
    SAM(rho=0.05).apply(model_sam, loss_fn_sam, batch_sam)
    sam_grad = T.cat([p.grad.view(-1).clone() for p in model_sam.parameters()])

    model_asam = _make_model()
    loss_fn_asam, batch_asam = _make_loss_fn(model_asam)
    loss = loss_fn_asam(*batch_asam)
    loss.backward()
    ASAM(rho=0.05).apply(model_asam, loss_fn_asam, batch_asam)
    asam_grad = T.cat([p.grad.view(-1).clone() for p in model_asam.parameters()])

    # Then ASAM and SAM produce different gradients
    assert not T.allclose(sam_grad, asam_grad, atol=1e-7)


# ---------------------------------------------------------------------------
# LAMP tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lamp_conforms_to_protocol() -> None:
    # Given a LAMPRollback instance
    lamp = LAMPRollback()

    # Then it satisfies the GradientTransform protocol
    assert isinstance(lamp, GradientTransform)


@pytest.mark.unit
def test_lamp_injects_noise() -> None:
    # Given a model with known parameters
    model = _make_model()
    params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When post_step is called
    LAMPRollback(eps=5e-3).post_step(model)

    # Then parameters differ from before (noise was injected)
    params_after = parameters_to_vector(model.parameters()).detach()
    assert not T.allclose(params_before, params_after, atol=1e-8)


@pytest.mark.unit
def test_lamp_rollback_resets_state() -> None:
    # Given a LAMPRollback with rollback_len=3
    lamp = LAMPRollback(eps=5e-3, rollback_len=3)
    model = _make_model()

    # When post_step is called rollback_len + 1 times (triggers rollback)
    params_before_rollback = parameters_to_vector(model.parameters()).detach().clone()
    for _ in range(4):
        lamp.post_step(model)
    params_after_rollback = parameters_to_vector(model.parameters()).detach()

    # Then internal state is reset and model params changed (rolled back to average)
    assert lamp.rollback_step == 0
    assert lamp.mean_params is not None
    assert T.allclose(lamp.mean_params, T.zeros_like(lamp.mean_params))
    assert not T.allclose(params_before_rollback, params_after_rollback, atol=1e-8)


@pytest.mark.unit
def test_lamp_rollback_step_increments() -> None:
    # Given a fresh LAMPRollback
    lamp = LAMPRollback(eps=5e-3, rollback_len=10)
    model = _make_model()

    # When post_step is called 3 times
    for _ in range(3):
        lamp.post_step(model)

    # Then rollback_step equals 3
    assert lamp.rollback_step == 3


# ---------------------------------------------------------------------------
# Composable pipeline test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sam_then_lamp_compose() -> None:
    # Given a model, SAM transform, LAMP transform, and an optimizer
    model = _make_model()
    loss_fn, batch = _make_loss_fn(model)
    optimizer = T.optim.SGD(model.parameters(), lr=1e-2)
    sam = SAM(rho=0.05)
    lamp = LAMPRollback(eps=5e-3, rollback_len=5)

    params_before = parameters_to_vector(model.parameters()).detach().clone()

    # When we run the full pipeline: forward, backward, SAM.apply, step, LAMP.post_step
    optimizer.zero_grad()
    loss = loss_fn(*batch)
    loss.backward()
    sam.apply(model, loss_fn, batch)
    optimizer.step()
    lamp.post_step(model)

    # Then parameters have changed (the pipeline ran without error)
    params_after = parameters_to_vector(model.parameters()).detach()
    assert not T.allclose(params_before, params_after, atol=1e-8)
