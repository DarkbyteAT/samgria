"""samgria — Composable gradient transforms for PyTorch — SAM, ASAM, LAMP, and beyond."""

from samgria.meta.maml import FOMAML as FOMAML
from samgria.meta.maml import MAML as MAML
from samgria.meta.protocol import MetaOptimizer as MetaOptimizer
from samgria.meta.protocol import mutation_optimizer as mutation_optimizer
from samgria.meta.protocol import sgd as sgd
from samgria.meta.reptile import Reptile as Reptile
from samgria.meta.step import MetaStep as MetaStep
from samgria.meta.step import meta_step as meta_step
from samgria.state import AdaptedState as AdaptedState
from samgria.state import ParameterSnapshot as ParameterSnapshot
from samgria.state import query_forward as query_forward
from samgria.state import restore_state as restore_state
from samgria.state import save_state as save_state
from samgria.transforms.asam import ASAM as ASAM
from samgria.transforms.lamp import LAMPRollback as LAMPRollback
from samgria.transforms.protocol import GradientTransform as GradientTransform
from samgria.transforms.sam import SAM as SAM
from samgria.utils.functional import functional_forward as functional_forward
from samgria.utils.grad import get_grad as get_grad
from samgria.utils.grad import set_grad as set_grad


__all__ = [
    "ASAM",
    "AdaptedState",
    "FOMAML",
    "GradientTransform",
    "LAMPRollback",
    "MAML",
    "MetaOptimizer",
    "MetaStep",
    "ParameterSnapshot",
    "Reptile",
    "SAM",
    "functional_forward",
    "get_grad",
    "meta_step",
    "mutation_optimizer",
    "query_forward",
    "sgd",
    "restore_state",
    "save_state",
    "set_grad",
]
