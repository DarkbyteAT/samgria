"""Complete model+optimizer state capture and restore.

Provides an immutable snapshot of the full training state: parameters,
optimizer buffers (e.g. Adam momentum/variance), and module buffers
(e.g. batch norm running statistics).  This generalises the parameter-only
save/restore used by SAM into a primitive suitable for meta-learning
inner loops, where multiple independent snapshots must coexist without
aliasing.

Typical usage::

    snapshot = save_state(model, optimizer)

    # ... run inner-loop adaptation (modifies model + optimizer) ...

    restore_state(model, optimizer, snapshot)  # back to outer-loop state

Design decisions:

- **Frozen dataclass** — snapshots are values, not mutable references.
  Multiple snapshots (one per task in MAML) are safe to hold simultaneously.
- **Detached + cloned tensors** — severs the autograd graph so snapshots
  never leak gradients into the outer optimisation.
- **Deep-copied optimizer state** — Adam's per-parameter momentum and
  variance buffers are independent copies, preventing silent corruption
  when the live optimizer continues training after a snapshot is taken.
- **Named buffers** — captures non-parameter state like batch norm
  ``running_mean`` / ``running_var``, which shift between tasks and
  cause silent performance degradation if not explicitly managed.

.. note::

   Not thread-safe.  Callers must ensure no concurrent forward/backward
   passes are running on the model or optimizer during save or restore.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters


__all__ = [
    "AdaptedState",
    "ParameterSnapshot",
    "query_forward",
    "restore_state",
    "save_state",
]


@dataclass(frozen=True)
class ParameterSnapshot:
    """Immutable checkpoint of model parameters, optimizer state, and buffers.

    Attributes
    ----------
    params
        Flattened parameter vector, detached and cloned from the model.
    numel
        Total number of scalar parameters.  Used by ``restore_state``
        to validate that the target model has the same architecture.
    optim_state
        Deep copy of the optimizer's ``state_dict()``.
    buffers
        Named module buffers (e.g. ``running_mean``, ``running_var``,
        ``num_batches_tracked``), each detached and cloned.
    """

    params: T.Tensor
    numel: int
    optim_state: dict[str, Any]
    buffers: dict[str, T.Tensor]


@dataclass(frozen=True)
class AdaptedState:
    """State produced by inner-loop adaptation.

    Composes a detached ``ParameterSnapshot`` (for state management,
    restore, and Reptile-style interpolation) with an optional dict of
    graph-connected parameter tensors (for MAML's second-order outer
    update via ``functional_call``).

    Attributes
    ----------
    snapshot
        Immutable, detached checkpoint of the adapted model state.
    live_params
        Named parameter tensors that retain the computation graph
        through the inner-loop steps.  Present for second-order methods
        (MAML); ``None`` for first-order methods (FOMAML, Reptile).
    """

    snapshot: ParameterSnapshot
    live_params: dict[str, T.Tensor] | None = field(default=None)


def query_forward(
    model: nn.Module,
    adapted: AdaptedState,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a forward pass through adapted parameters.

    When ``adapted.live_params`` is populated (MAML), the forward pass
    is routed through ``torch.func.functional_call`` so the computation
    graph flows back through the inner-loop steps.  Otherwise falls back
    to a plain ``model()`` call using the model's **currently loaded**
    parameters — the caller must ensure these are the adapted params
    (e.g. via ``restore_state``) before calling this function.

    Parameters
    ----------
    model
        The model whose forward method to call.
    adapted
        The adapted state from ``MetaOptimizer.adapt()``.
    *args, **kwargs
        Forwarded to the model's forward method.
    """
    if adapted.live_params is not None:
        return functional_call(model, adapted.live_params, args, kwargs)
    return model(*args, **kwargs)


def save_state(model: nn.Module, optimizer: optim.Optimizer) -> ParameterSnapshot:
    """Capture a complete, immutable snapshot of model and optimizer state.

    Parameters
    ----------
    model
        The model whose parameters and buffers to capture.
    optimizer
        The optimizer whose internal state (momentum, variance, step
        counts) to capture.

    Returns
    -------
    ParameterSnapshot
        A frozen snapshot that will not be affected by subsequent
        training steps.
    """
    params = parameters_to_vector(model.parameters()).detach().clone()
    optim_state = copy.deepcopy(optimizer.state_dict())
    buffers = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
    }
    return ParameterSnapshot(
        params=params,
        numel=params.numel(),
        optim_state=optim_state,
        buffers=buffers,
    )


def restore_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    snapshot: ParameterSnapshot,
) -> None:
    """Restore model parameters, optimizer state, and buffers from a snapshot.

    Parameters
    ----------
    model
        The model to restore.  Its parameters, gradients, and registered
        buffers will be overwritten.
    optimizer
        The optimizer to restore.  Its internal state (momentum buffers,
        step counts) will be replaced with the snapshot's copy.
    snapshot
        The snapshot to restore from.  It is not modified.

    Notes
    -----
    Gradients are cleared (set to ``None``) on all parameters after
    restore.  This prevents stale gradients from a previous inner loop
    leaking into the next task.

    All named buffers are restored, including batch norm's
    ``num_batches_tracked``.  This is correct for MAML (full state
    reset between tasks) but may not be appropriate for Reptile, which
    interpolates parameters without restoring.  Meta-optimizers that
    do not need buffer restore should handle this upstream.
    """
    # Validate parameter count matches
    model_numel = sum(p.numel() for p in model.parameters())
    if model_numel != snapshot.numel:
        raise ValueError(
            f"Model has {model_numel} parameters but snapshot has {snapshot.numel}. "
            f"Cannot restore a snapshot onto a model with different architecture."
        )

    # Validate buffer keys match
    model_buffer_keys = set(name for name, _ in model.named_buffers())
    snapshot_buffer_keys = set(snapshot.buffers.keys())
    if model_buffer_keys != snapshot_buffer_keys:
        missing = snapshot_buffer_keys - model_buffer_keys
        extra = model_buffer_keys - snapshot_buffer_keys
        parts: list[str] = []
        if missing:
            parts.append(f"snapshot has buffers not in model: {missing}")
        if extra:
            parts.append(f"model has buffers not in snapshot: {extra}")
        raise ValueError(
            f"Buffer mismatch between model and snapshot. {'; '.join(parts)}"
        )

    # Restore parameters and clear stale gradients.  Without this,
    # gradients from a previous inner loop could leak into the next
    # task — especially dangerous for MAML with create_graph=True.
    vector_to_parameters(snapshot.params.clone(), model.parameters())
    for p in model.parameters():
        p.grad = None

    # Restore optimizer state (deep copy so the snapshot stays pristine)
    optimizer.load_state_dict(copy.deepcopy(snapshot.optim_state))

    # Restore module buffers
    named_buffers = dict(model.named_buffers())
    for name, saved_buf in snapshot.buffers.items():
        named_buffers[name].copy_(saved_buf)
