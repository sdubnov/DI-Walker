"""Rollout implementation for frozen controllers and disturbance protocols."""

from __future__ import annotations

import numpy as np

from .config import BODY, DT, FORCE_SCALE, TAU
from .controllers import CentralControl, Controller
from .faults import LimbFault
from .tasks import Task


def rollout(
    controller: Controller,
    task: Task,
    central_control: CentralControl | None = None,
    limb_fault: LimbFault | None = None,
    record: bool = True,
) -> dict[str, np.ndarray]:
    """Simulate one controller/task/failure condition and return time-series logs."""

    central_control = central_control or CentralControl()
    limb_fault = limb_fault or LimbFault()
    target = task.target_path()

    state = np.zeros(6, dtype=float)
    activation = np.zeros(4, dtype=float)
    sensed_force = np.zeros(4, dtype=float)
    eye_history: list[float] = []
    cc_rng = np.random.default_rng(central_control.seed)

    rec = {
        "state": [],
        "act": [],
        "control": [],
        "force": [],
        "target": [],
        "error": [],
        "path_error": [],
        "cc": [],
    }

    for t in range(task.n_steps):
        ph, vx, vy, omega = state[2], state[3], state[4], state[5]
        c, sn = np.cos(ph), np.sin(ph)
        vf = vx * c + vy * sn
        vl = -vx * sn + vy * c
        ev = task.vd[t] - vf
        ew = task.wd[t] - omega

        q = controller.logits(task.vd[t], task.wd[t], state, activation, sensed_force)
        cc_delta, cc_scalar, raw_eye = central_control.correction(t, state, target[t], eye_history, cc_rng)
        eye_history.append(raw_eye)
        q = q + cc_delta

        u = np.tanh(q)
        activation += DT * (u - activation) / TAU
        activation = np.clip(activation, -1.0, 1.0)

        force = FORCE_SCALE * activation * limb_fault.multipliers(t)
        sensed_force = force.copy()

        total_force = force.sum()
        torque = np.sum(-BODY[:, 1] * force)
        state[3] += DT * (c * total_force - 0.85 * state[3])
        state[4] += DT * (sn * total_force - 0.85 * state[4])
        state[5] += DT * (torque / 0.18 - 0.60 * state[5])
        state[0] += DT * state[3]
        state[1] += DT * state[4]
        state[2] += DT * state[5]

        if record:
            rec["state"].append(state.copy())
            rec["act"].append(activation.copy())
            rec["control"].append(u.copy())
            rec["force"].append(force.copy())
            rec["target"].append(target[t].copy())
            rec["error"].append([ev, vl, ew])
            rec["path_error"].append(np.linalg.norm(state[:2] - target[t, :2]))
            rec["cc"].append(cc_scalar)

    return {k: np.asarray(v) for k, v in rec.items()}
