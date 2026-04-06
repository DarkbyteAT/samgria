"""Meta-learning optimizers — MAML, FOMAML, and Reptile.

Formalisms
----------

A standard learning problem optimises parameters theta over a single
dataset D and loss surface L:

    min_theta  (1/n) sum_i L(f_theta, D_i)

Meta-learning extends this to a *distribution of tasks*.  Each task
T_i = (D_i^s, D_i^q) has a support set (for adaptation) and a query
set (for evaluation).  The meta-objective over a batch of N tasks:

    min_theta  (1/N) sum_i w_i * L(f_{theta'_i}, D_i^q)

where w_i are per-task weights (uniform by default) and theta'_i is
the result of k inner gradient steps on the support set:

    theta_i^(0)   = theta
    theta_i^(j+1) = theta_i^(j) - alpha * T(grad L(f_{theta_i^(j)}, D_i^s))
    theta'_i      = theta_i^(k)

Here T is the composed gradient transform (identity by default; SAM,
ASAM, etc. when composed via ``grad_transforms``).

Algorithm variants differ in how the outer gradient is computed:

    MAML:   g = grad_theta (1/N) sum w_i * L(f_{theta'_i}, D_i^q)
            Full backprop through inner steps (second-order).

    FOMAML: g = (1/N) sum w_i * grad_{theta'_i} L(f_{theta'_i}, D_i^q)
            Treats adapted params as leaves (first-order approx).

    Reptile: g = (1/N) sum (theta'_i - theta)
             Parameter interpolation, no query set needed.
             theta <- theta + beta * g

Math-to-code mapping
--------------------

    Math                    Code
    ----                    ----
    theta                   model.parameters() at context entry
    D_i^s  (support)        support=(x_s, y_s)
    D_i^q  (query)          query=(x_q, y_q)
    w_i    (task weight)    weight=1.0  (kwarg on .task())
    k      (inner steps)    inner_steps=k
    T      (grad transform) grad_transforms=[SAM(...)]
    theta'_i = Adapt(...)   ms.task(...) -> AdaptedState
    L(f_{theta'_i}, D^q)    query loss (computed internally)
    Outer update            ms.step() / context manager __exit__

Extension points
----------------

- Per-task ``weight`` scales the query loss contribution.
- Per-task ``inner_steps`` and ``grad_transforms`` override defaults.
- ``inner_reg_fn`` adds regularisation to the inner loss (e.g. L2
  toward theta, KL divergence, elastic weight consolidation).
- ``query_loss_fn`` callback for custom query loss computation.
"""

from samgria.meta.maml import FOMAML as FOMAML
from samgria.meta.maml import MAML as MAML
from samgria.meta.protocol import MetaOptimizer as MetaOptimizer
from samgria.meta.reptile import Reptile as Reptile
from samgria.meta.step import MetaStep as MetaStep
from samgria.meta.step import meta_step as meta_step


__all__ = ["FOMAML", "MAML", "MetaOptimizer", "MetaStep", "Reptile", "meta_step"]
