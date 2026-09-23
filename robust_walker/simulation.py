"""Rollout implementation for frozen controllers and disturbance protocols."""

from __future__ import annotations

import numpy as np

from .config import BODY, DT, FORCE_SCALE, TAU
from .controllers import CentralControl, Controller
from .faults import ForceNoise, LimbFault, SensorFault
from .tasks import Task


def rollout(
    controller: Controller,
    task: Task,
    central_control: CentralControl | None = None,
    limb_fault: LimbFault | None = None,
    sensor_fault: SensorFault | None = None,
    force_noise: ForceNoise | None = None,
    sensor_bits: int | None = None,
    sensor_levels: int | None = None,
    external_force_y: np.ndarray | None = None,
    external_torque: np.ndarray | None = None,
    record: bool = True,
) -> dict[str, np.ndarray]:
    """Simulate one controller/task/failure condition and return time-series logs.

    ``external_force_y`` and ``external_torque`` are optional acceleration-scale
    disturbances indexed by simulation step. They affect the plant only; they
    are not supplied directly to the controller.
    """

    central_control = central_control or CentralControl()
    limb_fault = limb_fault or LimbFault()
    sensor_fault = sensor_fault or SensorFault()
    force_noise = force_noise or ForceNoise()
    if sensor_bits is not None and sensor_bits < 0:
        raise ValueError("sensor_bits must be nonnegative or None")
    if sensor_bits is not None and sensor_levels is not None:
        raise ValueError("provide sensor_bits or sensor_levels, not both")
    if sensor_levels is not None and (sensor_levels < 1 or sensor_levels % 2 == 0):
        raise ValueError("sensor_levels must be a positive odd integer")
    n_steps = task.n_steps
    external_force_y = np.zeros(n_steps, dtype=float) if external_force_y is None else np.asarray(external_force_y, dtype=float)
    external_torque = np.zeros(n_steps, dtype=float) if external_torque is None else np.asarray(external_torque, dtype=float)
    if external_force_y.shape != (n_steps,) or external_torque.shape != (n_steps,):
        raise ValueError("external disturbances must have one value per task step")
    target = task.target_path()

    state = np.zeros(6, dtype=float)
    activation = np.zeros(4, dtype=float)
    sensed_force = np.zeros(4, dtype=float)
    eye_history: list[float] = []
    cc_rng = np.random.default_rng(central_control.seed)

    rec = {
        "state": [],
        "state_before": [],
        "act": [],
        "act_before": [],
        "control": [],
        # Force-sensor values available to the controller before each update.
        "message": [],
        "message_available": [],
        "force": [],
        "target": [],
        "error": [],
        "path_error": [],
        "cc": [],
        "external_force_y": [],
        "external_torque": [],
        "message_clip_fraction": [],
    }

    for t in range(task.n_steps):
        message_available = sensor_fault.mask(t)
        if record:
            rec["state_before"].append(state.copy())
            rec["act_before"].append(activation.copy())
        ph, vx, vy, omega = state[2], state[3], state[4], state[5]
        c, sn = np.cos(ph), np.sin(ph)
        vf = vx * c + vy * sn
        vl = -vx * sn + vy * c
        ev = task.vd[t] - vf
        ew = task.wd[t] - omega

        sensed_for_control = sensed_force
        clipped_fraction = 0.0
        if sensor_bits is not None and sensor_bits >= 0:
            levels = max(1, 2**sensor_bits)
            normalized = np.clip(sensed_force / FORCE_SCALE, -1.0, 1.0)
            if levels == 1:
                sensed_for_control = np.zeros_like(sensed_force)
            else:
                step = 2.0 / (levels - 1)
                sensed_for_control = FORCE_SCALE * (np.round((normalized + 1.0) / step) * step - 1.0)
        elif sensor_levels is not None:
            normalized = sensed_force / FORCE_SCALE
            clipped_fraction = float(np.mean(np.abs(normalized) > 1.0))
            clipped = np.clip(normalized, -1.0, 1.0)
            if sensor_levels > 1:
                step = 2.0 / (sensor_levels - 1)
                sensed_for_control = FORCE_SCALE * (step * np.round(clipped / step))
            else:
                sensed_for_control = np.zeros_like(sensed_force)
        sensed_for_control = sensed_for_control * message_available
        if record:
            rec["message"].append(sensed_for_control.copy())
            rec["message_available"].append(message_available.copy())
            rec["message_clip_fraction"].append(clipped_fraction)
        q = controller.logits(task.vd[t], task.wd[t], state, activation, sensed_for_control)
        cc_delta, cc_scalar, raw_eye = central_control.correction(t, state, target[t], eye_history, cc_rng)
        eye_history.append(raw_eye)
        q = q + cc_delta

        u = np.tanh(q)
        activation += DT * (u - activation) / TAU
        activation = np.clip(activation, -1.0, 1.0)

        force = FORCE_SCALE * activation * limb_fault.multipliers(t)
        force *= force_noise.multipliers()
        sensed_force = force.copy() * sensor_fault.mask(t + 1)

        total_force = force.sum()
        torque = np.sum(-BODY[:, 1] * force)
        state[3] += DT * (c * total_force - 0.85 * state[3])
        state[4] += DT * (sn * total_force - 0.85 * state[4] + external_force_y[t])
        state[5] += DT * ((torque + external_torque[t]) / 0.18 - 0.60 * state[5])
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
            rec["external_force_y"].append(external_force_y[t])
            rec["external_torque"].append(external_torque[t])

    return {k: np.asarray(v) for k, v in rec.items()}
