"""Repository-ready robust quadruped simulation package."""

from .config import BODY, DT, N_LIMBS, TAU, T
from .controllers import Controller, ControllerParams, Regime
from .simulation import rollout
from .tasks import make_task

__all__ = [
    "BODY",
    "DT",
    "N_LIMBS",
    "TAU",
    "T",
    "Controller",
    "ControllerParams",
    "Regime",
    "make_task",
    "rollout",
]
