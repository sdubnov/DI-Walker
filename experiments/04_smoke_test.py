#!/usr/bin/env python3
"""Fast deterministic smoke test for the refactored rollout path."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import Controller
from robust_walker.metrics import tracking_loss
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


def main() -> None:
    params = np.zeros(36)
    controller = Controller.from_vector("capacity", params)
    result = rollout(controller, make_task("straight"))
    assert result["state"].shape == (240, 6)
    assert result["act"].shape == (240, 4)
    assert np.isfinite(tracking_loss(result))
    print("smoke test passed")


if __name__ == "__main__":
    main()
