"""Behaviour-driven tests for the functional_forward context manager.

These tests document the contract that functional_forward must satisfy:
    - Inside the context, model(x) uses the provided params dict.
    - Outside the context, model(x) uses stored parameters.
    - No infinite recursion (functional_call re-entry is handled).
    - The context manager is exception-safe (forward is always restored).
"""

from __future__ import annotations

import pytest
import torch as T
import torch.nn as nn

from samgria.utils.functional import functional_forward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlp() -> nn.Sequential:
    """Two-layer MLP with deterministic weights."""
    T.manual_seed(42)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_functional_forward_uses_provided_params() -> None:
    """Inside the context, model(x) routes through the provided params."""
    # Given a model and a zeroed-out params dict
    model = _make_mlp()
    zero_params = {k: T.zeros_like(v) for k, v in model.named_parameters()}
    x = T.randn(3, 2)

    # When we call model inside functional_forward with zero params
    with functional_forward(model, zero_params):
        output = model(x)

    # Then the output is all zeros (zero weights + zero bias = zero)
    assert T.equal(output, T.zeros(3, 1))


@pytest.mark.unit
def test_forward_restored_after_context_exits() -> None:
    """After the context exits, model(x) uses stored parameters again."""
    # Given a model
    model = _make_mlp()
    x = T.randn(3, 2)

    # And the normal output
    with T.no_grad():
        normal_output = model(x).clone()

    # When we use functional_forward then exit
    zero_params = {k: T.zeros_like(v) for k, v in model.named_parameters()}
    with functional_forward(model, zero_params):
        pass

    # Then model(x) is back to normal
    with T.no_grad():
        after_output = model(x)
    assert T.equal(after_output, normal_output)


@pytest.mark.unit
def test_no_recursion_with_functional_call() -> None:
    """functional_forward does not recurse when functional_call calls model."""
    # Given a model and params
    model = _make_mlp()
    params: dict[str, T.Tensor] = dict(model.named_parameters())
    x = T.randn(3, 2)

    # When we call model inside the context — this would infinitely recurse
    # if the un-patch/re-patch mechanism didn't work
    with functional_forward(model, params):
        output = model(x)

    # Then we get a result (no RecursionError)
    assert output.shape == (3, 1)


@pytest.mark.unit
def test_exception_safety_restores_forward() -> None:
    """forward is restored even if an exception occurs inside the context."""
    # Given a model and its normal output
    model = _make_mlp()
    x = T.randn(3, 2)
    with T.no_grad():
        normal_output = model(x).clone()

    # When an exception is raised inside the context
    zero_params = {k: T.zeros_like(v) for k, v in model.named_parameters()}
    with pytest.raises(ValueError, match="test"):
        with functional_forward(model, zero_params):
            raise ValueError("test")

    # Then model(x) produces normal output again (forward was restored)
    with T.no_grad():
        after_output = model(x)
    assert T.equal(after_output, normal_output)


@pytest.mark.unit
def test_graph_flows_through_functional_forward() -> None:
    """Gradients flow through functional_forward's output."""
    # Given a model and graph-connected params
    model = _make_mlp()
    params = {k: v.clone().requires_grad_(True) for k, v in model.named_parameters()}
    x = T.randn(3, 2)

    # When we compute a loss inside the context
    with functional_forward(model, params):
        output = model(x)
    loss = output.sum()

    # Then we can compute gradients w.r.t. the params via autograd.grad
    param_values = list(params.values())
    grads = T.autograd.grad(loss, param_values)
    for name, g in zip(params.keys(), grads, strict=True):
        assert g is not None, f"No gradient for {name}"
        assert g.shape == params[name].shape
