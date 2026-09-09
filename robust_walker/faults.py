"""Limb failure and intermittent disturbance models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LimbFault:
    """A retained-strength failure or intermittent slip applied to one limb."""

    limb: int | None = None
    mode: str = "intact"
    strength: float = 1.0
    start_step: int = 80
    slip_period: int = 24
    slip_steps: int = 10
    slip_strength: float = 0.2

    def multipliers(self, t: int, n_limbs: int = 4) -> np.ndarray:
        """Return per-limb force multipliers for simulation step t."""

        eff = np.ones(n_limbs, dtype=float)
        if self.limb is None or t < self.start_step or self.mode == "intact":
            return eff
        if self.mode == "loss":
            eff[self.limb] = self.strength
        elif self.mode == "slip":
            if ((t - self.start_step) % self.slip_period) < self.slip_steps:
                eff[self.limb] = self.slip_strength
        else:
            raise ValueError(f"unknown limb fault mode: {self.mode}")
        return eff


@dataclass(frozen=True)
class SensorFault:
    """Permanent loss of one sensor channel after a specified step."""

    limb: int | None = None
    start_step: int = 80

    def mask(self, t: int, n_limbs: int = 4) -> np.ndarray:
        """Return availability of each realized-force sensor at step t."""

        available = np.ones(n_limbs, dtype=float)
        if self.limb is not None and t >= self.start_step:
            available[self.limb] = 0.0
        return available


@dataclass
class ForceNoise:
    """Independent multiplicative outcome noise applied to each limb force."""

    sigma: float = 0.0
    correlation: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError("correlation must be in [0, 1)")
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros(4, dtype=float)

    def multipliers(self, n_limbs: int = 4) -> np.ndarray:
        """Return independent, optionally temporally correlated force multipliers."""

        white = self._rng.normal(size=n_limbs)
        self._state = self.correlation * self._state + np.sqrt(1.0 - self.correlation**2) * white
        return np.clip(1.0 + self.sigma * self._state, 0.0, None)
