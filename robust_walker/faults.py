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
