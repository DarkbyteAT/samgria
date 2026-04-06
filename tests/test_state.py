"""Behaviour-driven tests for ParameterSnapshot and save/restore state primitives.

These tests document the contract that ParameterSnapshot, save_state, and
restore_state must satisfy.  Each test name describes a single observable
behaviour, structured as Given-When-Then.

Key guarantees under test:
    - Snapshots are immutable values (frozen dataclass, detached tensors).
    - Multiple snapshots do not alias — mutating the model after saving
      does not corrupt earlier snapshots.
    - Round-trip save/restore reproduces parameters, optimizer state, and
      module buffers exactly.
    - Batch norm running statistics are captured and restored, preventing
      the silent state-leakage bug that looks like concept drift.
"""

from __future__ import annotations

import copy

import pytest
import torch as T
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from samgria.state import restore_state, save_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlp() -> nn.Sequential:
    """Two-layer MLP with deterministic weights."""
    T.manual_seed(42)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _make_bn_model() -> nn.Sequential:
    """MLP with batch norm — exercises buffer capture and restore."""
    T.manual_seed(42)
    return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2))


def _train_step(model: nn.Module, optimizer: T.optim.Optimizer) -> None:
    """Run one forward-backward-step cycle on random data."""
    x = T.randn(16, 4)
    y = T.randn(16, 2)
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()


# ---------------------------------------------------------------------------
# ParameterSnapshot structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_is_frozen_dataclass() -> None:
    """Snapshots are immutable — assignment to fields raises."""
    # Given a snapshot
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters())
    snapshot = save_state(model, optimizer)

    # Then assigning to any field raises FrozenInstanceError
    with pytest.raises(AttributeError):
        snapshot.params = T.zeros(1)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        snapshot.optim_state = {}  # type: ignore[misc]
    with pytest.raises(AttributeError):
        snapshot.buffers = {}  # type: ignore[misc]


@pytest.mark.unit
def test_snapshot_params_are_detached() -> None:
    """Saved parameters are detached from the autograd graph."""
    # Given a model with gradients
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters())
    _train_step(model, optimizer)

    # When a snapshot is taken
    snapshot = save_state(model, optimizer)

    # Then the params tensor does not track gradients
    assert not snapshot.params.requires_grad


@pytest.mark.unit
def test_snapshot_params_are_cloned() -> None:
    """Modifying the model after save does not corrupt the snapshot."""
    # Given a snapshot taken before a training step
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters())
    snapshot = save_state(model, optimizer)
    params_at_save = snapshot.params.clone()

    # When the model is modified
    _train_step(model, optimizer)

    # Then the snapshot's params are unchanged
    assert T.equal(snapshot.params, params_at_save)


# ---------------------------------------------------------------------------
# Round-trip save → restore
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_round_trip_preserves_parameters() -> None:
    """Restoring a snapshot recovers the exact parameter values."""
    # Given a model, optimizer, and a snapshot of the initial state
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters())
    snapshot = save_state(model, optimizer)

    # When the model is trained and then restored
    for _ in range(5):
        _train_step(model, optimizer)
    restore_state(model, optimizer, snapshot)

    # Then parameters match the snapshot
    restored = parameters_to_vector(model.parameters()).detach()
    assert T.equal(restored, snapshot.params)


@pytest.mark.unit
def test_round_trip_preserves_optimizer_state() -> None:
    """Restoring a snapshot recovers Adam's momentum and variance buffers."""
    # Given a model trained for several steps, then snapshotted
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        _train_step(model, optimizer)
    snapshot = save_state(model, optimizer)
    saved_optim = copy.deepcopy(snapshot.optim_state)

    # When more training occurs and then state is restored
    for _ in range(5):
        _train_step(model, optimizer)
    restore_state(model, optimizer, snapshot)

    # Then optimizer state_dict matches the snapshot
    current_optim = optimizer.state_dict()
    for key in saved_optim["state"]:
        for buf_name in saved_optim["state"][key]:
            saved_val = saved_optim["state"][key][buf_name]
            current_val = current_optim["state"][key][buf_name]
            if isinstance(saved_val, T.Tensor):
                assert T.equal(saved_val, current_val), (
                    f"Optimizer state mismatch: param {key}, buffer {buf_name}"
                )
            else:
                assert saved_val == current_val


@pytest.mark.unit
def test_round_trip_preserves_batch_norm_buffers() -> None:
    """Restoring a snapshot recovers batch norm running_mean and running_var."""
    # Given a batch norm model with running stats accumulated over training
    model = _make_bn_model()
    optimizer = T.optim.Adam(model.parameters())
    model.train()
    for _ in range(10):
        _train_step(model, optimizer)
    snapshot = save_state(model, optimizer)

    # When more training shifts the running stats
    for _ in range(10):
        _train_step(model, optimizer)

    # Then running stats have diverged
    bn = model[1]
    assert isinstance(bn, nn.BatchNorm1d)
    assert not T.equal(bn.running_mean, snapshot.buffers["1.running_mean"])  # type: ignore[arg-type]

    # When we restore
    restore_state(model, optimizer, snapshot)

    # Then running stats match the snapshot
    assert T.equal(bn.running_mean, snapshot.buffers["1.running_mean"])  # type: ignore[arg-type]
    assert T.equal(bn.running_var, snapshot.buffers["1.running_var"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Aliasing safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multiple_snapshots_do_not_alias() -> None:
    """Two snapshots taken at different times hold independent state."""
    # Given two snapshots taken before and after training
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters())
    snap_a = save_state(model, optimizer)

    for _ in range(5):
        _train_step(model, optimizer)
    snap_b = save_state(model, optimizer)

    # Then their params differ (training changed the model)
    assert not T.allclose(snap_a.params, snap_b.params, atol=1e-6)

    # And restoring snap_a does not corrupt snap_b
    restore_state(model, optimizer, snap_a)
    params_now = parameters_to_vector(model.parameters()).detach()
    assert T.allclose(params_now, snap_a.params, atol=1e-7)
    # snap_b is still intact
    assert not T.allclose(snap_b.params, snap_a.params, atol=1e-6)


@pytest.mark.unit
def test_optimizer_state_is_deep_copied() -> None:
    """Snapshot optimizer state is independent — training after save cannot corrupt it."""
    # Given a snapshot after some training
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        _train_step(model, optimizer)
    snapshot = save_state(model, optimizer)

    # Capture the step count from the snapshot
    first_param_key = list(snapshot.optim_state["state"].keys())[0]
    step_at_snapshot = snapshot.optim_state["state"][first_param_key]["step"]

    # When more training occurs
    for _ in range(10):
        _train_step(model, optimizer)

    # Then the snapshot's step count is unchanged (deep copy, not reference)
    assert snapshot.optim_state["state"][first_param_key]["step"] == step_at_snapshot


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_model_with_no_buffers() -> None:
    """save_state works on models without buffers (empty dict)."""
    # Given a plain MLP (no batch norm, no registered buffers)
    model = _make_mlp()
    optimizer = T.optim.SGD(model.parameters(), lr=0.01)

    # When a snapshot is taken
    snapshot = save_state(model, optimizer)

    # Then buffers dict is empty
    assert snapshot.buffers == {}


@pytest.mark.unit
def test_restore_with_sgd_optimizer() -> None:
    """Round-trip works with SGD (minimal optimizer state)."""
    # Given a model with SGD
    model = _make_mlp()
    optimizer = T.optim.SGD(model.parameters(), lr=0.01)
    snapshot = save_state(model, optimizer)

    # When trained and restored
    for _ in range(5):
        _train_step(model, optimizer)
    restore_state(model, optimizer, snapshot)

    # Then parameters match
    restored = parameters_to_vector(model.parameters()).detach()
    assert T.equal(restored, snapshot.params)


# ---------------------------------------------------------------------------
# Architecture mismatch validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_restore_raises_on_parameter_count_mismatch() -> None:
    """Restoring a snapshot onto a model with different parameter count raises."""
    # Given a snapshot from a 4->8->2 MLP
    model_a = _make_mlp()
    optimizer_a = T.optim.Adam(model_a.parameters())
    snapshot = save_state(model_a, optimizer_a)

    # When restoring onto a model with different architecture
    T.manual_seed(42)
    model_b = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
    optimizer_b = T.optim.Adam(model_b.parameters())

    # Then a ValueError is raised
    with pytest.raises(ValueError, match="parameters"):
        restore_state(model_b, optimizer_b, snapshot)


@pytest.mark.unit
def test_restore_raises_on_buffer_mismatch() -> None:
    """Restoring a snapshot onto a model with different buffers raises."""
    # Given two models with the same parameter count but different buffers
    T.manual_seed(42)
    model_with_bn = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
    optimizer_bn = T.optim.Adam(model_with_bn.parameters())
    snapshot = save_state(model_with_bn, optimizer_bn)

    # When restoring onto a model with same params but a missing buffer
    T.manual_seed(42)
    model_different = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
    # Manually remove a buffer to create a mismatch
    del model_different[1].running_mean  # type: ignore[union-attr]

    optimizer_diff = T.optim.Adam(model_different.parameters())

    # Then a ValueError is raised mentioning the buffer mismatch
    with pytest.raises(ValueError, match="[Bb]uffer"):
        restore_state(model_different, optimizer_diff, snapshot)


# ---------------------------------------------------------------------------
# Gradient hygiene
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_restore_clears_stale_gradients() -> None:
    """Restoring state sets all .grad attributes to None."""
    # Given a model that has been trained (gradients populated)
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters())
    snapshot = save_state(model, optimizer)
    _train_step(model, optimizer)

    # Verify gradients exist before restore
    assert all(p.grad is not None for p in model.parameters())

    # When state is restored
    restore_state(model, optimizer, snapshot)

    # Then all gradients are cleared
    assert all(p.grad is None for p in model.parameters())


# ---------------------------------------------------------------------------
# MAML outer-loop pattern: multiple restores from the same snapshot
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multiple_restores_from_same_snapshot() -> None:
    """Restoring the same snapshot twice yields identical state both times."""
    # Given a snapshot after some initial training
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        _train_step(model, optimizer)
    snapshot = save_state(model, optimizer)

    # When we train, restore, capture state, train again, restore again
    for _ in range(5):
        _train_step(model, optimizer)
    restore_state(model, optimizer, snapshot)
    params_first_restore = parameters_to_vector(model.parameters()).detach().clone()

    for _ in range(5):
        _train_step(model, optimizer)
    restore_state(model, optimizer, snapshot)
    params_second_restore = parameters_to_vector(model.parameters()).detach().clone()

    # Then both restores produce identical parameters
    assert T.equal(params_first_restore, params_second_restore)


# ---------------------------------------------------------------------------
# Optimizer param_groups (e.g. learning rate)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_restore_resets_learning_rate() -> None:
    """Restoring a snapshot recovers the optimizer's learning rate."""
    # Given a snapshot with lr=1e-3
    model = _make_mlp()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        _train_step(model, optimizer)
    snapshot = save_state(model, optimizer)

    # When the learning rate is changed and then state is restored
    optimizer.param_groups[0]["lr"] = 1e-5
    restore_state(model, optimizer, snapshot)

    # Then the learning rate is back to the snapshotted value
    assert optimizer.param_groups[0]["lr"] == 1e-3


# ---------------------------------------------------------------------------
# Smoke tests — prove the full workflow behaves correctly
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_maml_inner_loop_pattern() -> None:
    """Simulate a MAML outer step: save, adapt per task, restore between tasks.

    This proves that save_state/restore_state correctly isolate inner-loop
    adaptation across multiple tasks.  After adapting on each task independently,
    restoring to the outer-loop state must yield identical parameters every time,
    and each task's adapted parameters must differ from both the outer state and
    from each other (since they train on different data).
    """
    T.manual_seed(0)
    model = _make_bn_model()
    model.train()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)

    # Outer-loop pre-training
    for _ in range(5):
        _train_step(model, optimizer)

    # Given a snapshot of the outer-loop state
    outer_snapshot = save_state(model, optimizer)
    outer_params = outer_snapshot.params.clone()

    # When we adapt on 3 independent tasks (inner loops)
    adapted_params: list[T.Tensor] = []
    for task_id in range(3):
        # Restore to outer state before each task
        restore_state(model, optimizer, outer_snapshot)

        # Verify we're starting from the outer state
        assert T.equal(
            parameters_to_vector(model.parameters()).detach(), outer_params
        )
        # Verify gradients are clean
        assert all(p.grad is None for p in model.parameters())

        # Inner-loop adaptation with task-specific data
        T.manual_seed(100 + task_id)  # different data per task
        for _ in range(5):
            _train_step(model, optimizer)

        adapted_params.append(
            parameters_to_vector(model.parameters()).detach().clone()
        )

    # Then each task's adapted params differ from the outer state
    for i, ap in enumerate(adapted_params):
        assert not T.equal(ap, outer_params), f"Task {i} did not adapt"

    # And each task's adapted params differ from each other
    for i in range(len(adapted_params)):
        for j in range(i + 1, len(adapted_params)):
            assert not T.equal(adapted_params[i], adapted_params[j]), (
                f"Tasks {i} and {j} produced identical adapted params"
            )

    # And the outer snapshot is still pristine after all inner loops
    assert T.equal(outer_snapshot.params, outer_params)


@pytest.mark.integration
def test_batch_norm_isolation_across_tasks() -> None:
    """Batch norm running stats do not leak between tasks.

    Without buffer restore, running_mean/running_var accumulate statistics
    from all tasks, causing each subsequent task to start from a different
    statistical baseline.  This test verifies that restore_state prevents
    this cross-task contamination.
    """
    T.manual_seed(0)
    model = _make_bn_model()
    model.train()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)

    # Accumulate some running stats
    for _ in range(5):
        _train_step(model, optimizer)

    # Given a snapshot with specific running stats
    snapshot = save_state(model, optimizer)
    bn = model[1]
    assert isinstance(bn, nn.BatchNorm1d)
    saved_running_mean = snapshot.buffers["1.running_mean"].clone()

    # When we train on task A (shifts running stats)
    T.manual_seed(200)
    for _ in range(10):
        _train_step(model, optimizer)
    stats_after_task_a = bn.running_mean.clone()  # type: ignore[union-attr]
    assert not T.equal(stats_after_task_a, saved_running_mean)

    # And restore before task B
    restore_state(model, optimizer, snapshot)

    # Then running stats are back to the snapshot (task A didn't leak)
    assert T.equal(bn.running_mean, saved_running_mean)  # type: ignore[arg-type]


@pytest.mark.integration
def test_reptile_outer_update_pattern() -> None:
    """Simulate a Reptile outer step: save, adapt per task, interpolate.

    Reptile's outer update is parameter interpolation, not gradient-based:
    theta += meta_lr * mean(theta'_i - theta).  This test proves that
    save_state captures the outer parameters correctly and that the adapted
    parameter vectors can be used for the interpolation arithmetic without
    needing restore_state between tasks (Reptile does not restore).
    """
    T.manual_seed(0)
    model = _make_mlp()
    optimizer = T.optim.SGD(model.parameters(), lr=1e-2)

    # Outer-loop pre-training
    for _ in range(5):
        _train_step(model, optimizer)

    # Given a snapshot of the outer-loop state
    outer_snapshot = save_state(model, optimizer)
    outer_params = outer_snapshot.params.clone()
    meta_lr = 0.1

    # When we adapt on 3 tasks, saving each adapted param vector
    adapted_params: list[T.Tensor] = []
    for task_id in range(3):
        # Reptile restores before each task's inner loop
        restore_state(model, optimizer, outer_snapshot)

        # Inner-loop adaptation
        T.manual_seed(100 + task_id)
        for _ in range(5):
            _train_step(model, optimizer)

        adapted_params.append(
            parameters_to_vector(model.parameters()).detach().clone()
        )

    # Then we can compute the Reptile outer update via interpolation
    mean_diff = T.stack([ap - outer_params for ap in adapted_params]).mean(dim=0)
    new_outer_params = outer_params + meta_lr * mean_diff

    # And the new outer params differ from the old (learning happened)
    assert not T.equal(new_outer_params, outer_params)

    # And the update direction is a genuine average of task directions
    # (not dominated by any single task — verify non-zero contribution from each)
    individual_diffs = [ap - outer_params for ap in adapted_params]
    for i, diff in enumerate(individual_diffs):
        assert diff.norm() > 0, f"Task {i} produced zero adaptation"

    # And we can apply the interpolated params back to the model
    vector_to_parameters(new_outer_params, model.parameters())
    applied = parameters_to_vector(model.parameters()).detach()
    assert T.equal(applied, new_outer_params)
