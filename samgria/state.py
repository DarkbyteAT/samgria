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
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters


__all__ = ["ParameterSnapshot", "restore_state", "save_state"]


@dataclass(frozen=True)
class ParameterSnapshot:
    """Immutable checkpoint of model parameters, optimizer state, and buffers.

    Attributes
    ----------
    params
        Flattened parameter vector, detached and cloned from the model.
    optim_state
        Deep copy of the optimizer's ``state_dict()``.
    buffers
        Named module buffers (e.g. ``running_mean``, ``running_var``),
        each detached and cloned.
    """

    params: T.Tensor
    optim_state: dict[str, Any]
    buffers: dict[str, T.Tensor]


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
    return ParameterSnapshot(params=params, optim_state=optim_state, buffers=buffers)


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
    """
    # Restore parameters
    vector_to_parameters(snapshot.params.clone(), model.parameters())

    # Restore optimizer state (deep copy so the snapshot stays pristine)
    optimizer.load_state_dict(copy.deepcopy(snapshot.optim_state))

    # Restore module buffers
    named_buffers = dict(model.named_buffers())
    for name, saved_buf in snapshot.buffers.items():
        if name in named_buffers:
            named_buffers[name].copy_(saved_buf)
