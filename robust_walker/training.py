"""Vectorized CEM training for the matched-capacity controller regimes."""

from __future__ import annotations

import numpy as np

from .config import BODY, DT, FORCE_SCALE, PARAM_DIM, T, TAU
from .controllers import Regime
from .faults import LimbFault
from .tasks import TRAIN_TASKS, Task

TRAIN_SCENARIOS: list[tuple[Task, LimbFault]] = [(task, LimbFault()) for task in TRAIN_TASKS]
for limb, task in enumerate([TRAIN_TASKS[0], TRAIN_TASKS[2], TRAIN_TASKS[4], TRAIN_TASKS[6]]):
    TRAIN_SCENARIOS.append((task, LimbFault(limb=limb, mode="loss", strength=0.75, start_step=80)))


def population_losses(params: np.ndarray, regime: str | Regime, scenarios=TRAIN_SCENARIOS) -> np.ndarray:
    """Evaluate a candidate population on the V4.1b training curriculum."""

    regime = Regime(regime)
    pop = params.shape[0]
    bias, gv, gcmd, gyaw, glat, gself = [params[:, k : k + 4] for k in range(0, 24, 4)]
    weights = params[:, 24:].reshape(pop, 4, 3)
    losses = []

    for task, fault in scenarios:
        state = np.zeros((pop, 6), dtype=float)
        activation = np.zeros((pop, 4), dtype=float)
        sensed_force = np.zeros((pop, 4), dtype=float)
        desired_heading = np.cumsum(task.wd) * DT
        speed_err = np.zeros(pop)
        lateral_err = np.zeros(pop)
        yaw_err = np.zeros(pop)
        act_cost = np.zeros(pop)
        heading_err = np.zeros(pop)
        n = 0

        for t in range(T):
            ph = state[:, 2]
            vx = state[:, 3]
            vy = state[:, 4]
            omega = state[:, 5]
            c = np.cos(ph)
            sn = np.sin(ph)
            vf = vx * c + vy * sn
            vl = -vx * sn + vy * c
            ev = task.vd[t] - vf
            ew = task.wd[t] - omega

            q = (
                bias
                + gv * ev[:, None]
                + gcmd * task.wd[t]
                + gyaw * ew[:, None]
                - glat * vl[:, None]
                + gself * activation
            )
            if regime == Regime.CAPACITY:
                own = np.stack((activation, activation * activation, np.tanh(2 * activation)), axis=2)
                q += np.sum(weights * own, axis=2)
            elif regime == Regime.ACT_COMM:
                other = np.empty((pop, 4, 3), dtype=float)
                for i in range(4):
                    other[:, i, :] = np.delete(activation, i, axis=1)
                q += np.sum(weights * other, axis=2)
            elif regime == Regime.SENSOR_COMM:
                source = sensed_force / FORCE_SCALE
                other = np.empty((pop, 4, 3), dtype=float)
                for i in range(4):
                    other[:, i, :] = np.delete(source, i, axis=1)
                q += np.sum(weights * other, axis=2)
            else:
                raise ValueError(f"unknown regime: {regime}")

            control = np.tanh(q)
            activation += DT * (control - activation) / TAU
            activation = np.clip(activation, -1.0, 1.0)

            force = FORCE_SCALE * activation.copy()
            if fault.limb is not None and t >= fault.start_step:
                if fault.mode == "loss":
                    force[:, fault.limb] *= fault.strength
                elif fault.mode == "slip" and ((t - fault.start_step) % fault.slip_period) < fault.slip_steps:
                    force[:, fault.limb] *= fault.slip_strength
            sensed_force = force.copy()

            total_force = force.sum(axis=1)
            torque = np.sum((-BODY[:, 1])[None, :] * force, axis=1)
            state[:, 3] += DT * (c * total_force - 0.85 * state[:, 3])
            state[:, 4] += DT * (sn * total_force - 0.85 * state[:, 4])
            state[:, 5] += DT * (torque / 0.18 - 0.60 * state[:, 5])
            state[:, 0] += DT * state[:, 3]
            state[:, 1] += DT * state[:, 4]
            state[:, 2] += DT * state[:, 5]

            if t >= 20:
                speed_err += ev * ev
                lateral_err += vl * vl
                yaw_err += ew * ew
                act_cost += np.mean(activation * activation, axis=1)
                heading_err += (state[:, 2] - desired_heading[t]) ** 2
                n += 1

        losses.append(
            1.5 * speed_err / n
            + lateral_err / n
            + 2.2 * yaw_err / n
            + 0.02 * act_cost / n
            + 0.20 * heading_err / n
        )

    return np.mean(np.stack(losses, axis=1), axis=1)


def cem(
    regime: str | Regime,
    seed: int,
    iters: int = 12,
    pop: int = 36,
    elite: int = 7,
    std: float = 0.40,
) -> tuple[np.ndarray, float]:
    """Fit one policy with the Cross-Entropy Method optimizer."""

    rng = np.random.default_rng(seed)
    mu = np.zeros(PARAM_DIM, dtype=float)
    sd = np.full(PARAM_DIM, std, dtype=float)
    best = mu.copy()
    best_loss = np.inf

    for _ in range(iters):
        candidates = mu + sd * rng.normal(size=(pop, PARAM_DIM))
        losses = population_losses(candidates, regime)
        ids = np.argsort(losses)[:elite]
        elites = candidates[ids]
        mu = 0.25 * mu + 0.75 * elites.mean(axis=0)
        sd = np.maximum(0.035, 0.25 * sd + 0.75 * elites.std(axis=0))
        if losses[ids[0]] < best_loss:
            best_loss = float(losses[ids[0]])
            best = candidates[ids[0]].copy()

    return best, best_loss
