"""Controller architectures and weak central-control channel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import BODY, FORCE_SCALE, PARAM_DIM


class Regime(str, Enum):
    """Controller information-routing regimes."""

    CAPACITY = "capacity"
    SENSOR_COMM = "sensor_comm"


@dataclass(frozen=True)
class ControllerParams:
    """Structured view of the 36-parameter policy vector."""

    bias: np.ndarray
    gv: np.ndarray
    gcmd: np.ndarray
    gyaw: np.ndarray
    glat: np.ndarray
    gself: np.ndarray
    W: np.ndarray

    @classmethod
    def from_vector(cls, p: np.ndarray) -> "ControllerParams":
        """Split a flat 36-vector into local gains and communication weights."""

        p = np.asarray(p, dtype=float)
        if p.shape != (PARAM_DIM,):
            raise ValueError(f"expected parameter vector of shape {(PARAM_DIM,)}, got {p.shape}")
        chunks = [p[k : k + 4] for k in range(0, 24, 4)]
        return cls(*chunks, W=p[24:].reshape(4, 3))


@dataclass(frozen=True)
class Controller:
    """Frozen limb controller with a selected peer-information architecture."""

    regime: Regime
    params: ControllerParams

    @classmethod
    def from_vector(cls, regime: str | Regime, p: np.ndarray) -> "Controller":
        """Build a controller from a regime name and flat parameter vector."""

        return cls(Regime(regime), ControllerParams.from_vector(p))

    def logits(
        self,
        vd: float,
        wd: float,
        body_state: np.ndarray,
        activation: np.ndarray,
        sensed_force: np.ndarray,
    ) -> np.ndarray:
        """Compute the four limb logits before actuator saturation/dynamics."""

        p = self.params
        ph, vx, vy, omega = body_state[2], body_state[3], body_state[4], body_state[5]
        c, sn = np.cos(ph), np.sin(ph)
        vf = vx * c + vy * sn
        vl = -vx * sn + vy * c
        ev = vd - vf
        ew = wd - omega
        q = p.bias + p.gv * ev + p.gcmd * wd + p.gyaw * ew - p.glat * vl + p.gself * activation

        if self.regime == Regime.CAPACITY:
            own_sensor = sensed_force / FORCE_SCALE
            own = np.column_stack((own_sensor, own_sensor * own_sensor, np.tanh(2 * own_sensor)))
            q += np.sum(p.W * own, axis=1)
        elif self.regime == Regime.SENSOR_COMM:
            source = sensed_force / FORCE_SCALE
            for i in range(4):
                q[i] += p.W[i] @ np.delete(source, i)
        else:
            raise ValueError(f"unknown regime: {self.regime}")

        return q


@dataclass(frozen=True)
class CentralControl:
    """Weak global error channel with optional disruption modes."""

    gain: float = 0.0
    mode: str = "none"
    drop_step: int = 120
    period: int = 30
    on_steps: int = 10
    delay_steps: int = 5
    noise_std: float = 0.15
    stuck_step: int = 120
    seed: int = 0

    def availability(self, t: int) -> float:
        """Return whether the central-control signal is available at step t."""

        if self.mode in {"always", "delayed", "noisy", "stuck"}:
            return 1.0
        if self.mode == "dropout":
            return float(t < self.drop_step)
        if self.mode == "intermittent":
            return float((t % self.period) < self.on_steps)
        if self.mode == "none":
            return 0.0
        raise ValueError(f"unknown central-control mode: {self.mode}")

    @staticmethod
    def raw_eye(state: np.ndarray, target: np.ndarray) -> float:
        """Compute the unclipped weak target-relative correction signal."""

        ex = target[0] - state[0]
        ey = target[1] - state[1]
        ph = state[2]
        e_lat = -np.sin(ph) * ex + np.cos(ph) * ey
        e_head = np.arctan2(np.sin(target[2] - ph), np.cos(target[2] - ph))
        return float(np.clip(0.65 * e_lat + 0.35 * e_head, -1.0, 1.0))

    def correction(
        self,
        t: int,
        state: np.ndarray,
        target: np.ndarray,
        eye_history: list[float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Return per-limb CC correction, scalar CC value, and raw eye signal."""

        raw_eye = self.raw_eye(state, target)
        eye = raw_eye
        if self.mode == "delayed":
            if eye_history is None or len(eye_history) <= self.delay_steps:
                eye = 0.0
            else:
                eye = eye_history[-self.delay_steps]
        elif self.mode == "noisy":
            if rng is None:
                rng = np.random.default_rng(self.seed)
            eye = float(np.clip(raw_eye + rng.normal(0.0, self.noise_std), -1.0, 1.0))
        elif self.mode == "stuck" and t >= self.stuck_step:
            if eye_history is not None and len(eye_history) > self.stuck_step:
                eye = eye_history[self.stuck_step]
        cc = self.gain * self.availability(t) * eye
        side = np.sign(-BODY[:, 1])
        return cc * side, float(cc), raw_eye
