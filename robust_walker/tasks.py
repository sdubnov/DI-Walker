"""Goal-conditioned velocity and yaw-rate tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DT, T


@dataclass(frozen=True)
class Task:
    """A commanded forward-velocity and yaw-rate trajectory."""

    name: str
    vd: np.ndarray
    wd: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(len(self.vd))

    def target_path(self) -> np.ndarray:
        """Integrate the commanded velocity/yaw sequence into an ideal path."""

        target = np.zeros((self.n_steps, 3), dtype=float)
        for t in range(1, self.n_steps):
            target[t, 2] = target[t - 1, 2] + DT * self.wd[t - 1]
            target[t, 0] = target[t - 1, 0] + DT * self.vd[t - 1] * np.cos(target[t - 1, 2])
            target[t, 1] = target[t - 1, 1] + DT * self.vd[t - 1] * np.sin(target[t - 1, 2])
        return target


def make_task(name: str, v: float = 1.0, w: float = 0.18, n_steps: int = T) -> Task:
    """Create one of the named benchmark trajectory tasks."""

    vd = np.full(n_steps, float(v), dtype=float)
    wd = np.zeros(n_steps, dtype=float)

    if name == "straight":
        pass
    elif name == "left":
        wd[80:160] = abs(w)
    elif name == "right":
        wd[80:160] = -abs(w)
    elif name == "s_lr":
        wd[55:115] = abs(w)
        wd[125:185] = -abs(w)
    elif name == "s_rl":
        wd[55:115] = -abs(w)
        wd[125:185] = abs(w)
    elif name == "speed":
        vd[:80] = 0.8 * v
        vd[80:160] = 1.2 * v
    elif name == "sine":
        tt = np.arange(n_steps) * DT
        wd = w * np.sin(2 * np.pi * tt / 6)
    elif name == "chirp":
        tt = np.arange(n_steps) * DT
        wd = w * np.sin(2 * np.pi * (0.05 * tt + 0.018 * tt**2))
    else:
        raise ValueError(f"unknown task: {name}")

    return Task(name=name, vd=vd, wd=wd)


TRAIN_TASKS = [
    make_task("straight", 0.9),
    make_task("straight", 1.1),
    make_task("left", 1.0, 0.12),
    make_task("left", 1.0, 0.24),
    make_task("right", 1.0, 0.12),
    make_task("right", 1.0, 0.24),
    make_task("speed", 1.0),
]

ID_TASKS = [
    make_task("straight", 1.0),
    make_task("left", 1.0, 0.18),
    make_task("right", 1.0, 0.18),
]

OOS_TASKS = [
    make_task("s_lr", 1.0, 0.18),
    make_task("s_rl", 1.0, 0.18),
    make_task("sine", 1.0, 0.18),
    make_task("chirp", 1.0, 0.18),
    make_task("left", 1.15, 0.27),
    make_task("right", 0.85, 0.27),
]
