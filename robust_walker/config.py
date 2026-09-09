"""Physical constants for the frozen quadruped model."""

from __future__ import annotations

import numpy as np

DT = 0.05
T = 240
N_LIMBS = 4
TAU = 0.22
FORCE_SCALE = 1.18
BODY = np.array(
    [
        [0.32, 0.22],
        [0.32, -0.22],
        [-0.32, 0.22],
        [-0.32, -0.22],
    ],
    dtype=float,
)
LOCAL_PARAM_DIM = 24
# Legacy matched-parameter policy dimension (Own-Kernel and Peer-Sensor).
PARAM_DIM = 36
