"""Functional tracking and robustness metrics."""

from __future__ import annotations

import numpy as np


def tracking_loss(rollout: dict[str, np.ndarray], burn_in: int = 20) -> float:
    """Compute the V4.1b functional tracking loss after an initial burn-in."""

    err = rollout["error"][burn_in:]
    act = rollout["act"][burn_in:]
    heading = rollout["state"][burn_in:, 2] - rollout["target"][burn_in:, 2]
    return float(
        1.5 * np.mean(err[:, 0] ** 2)
        + np.mean(err[:, 1] ** 2)
        + 2.2 * np.mean(err[:, 2] ** 2)
        + 0.02 * np.mean(act**2)
        + 0.20 * np.mean(heading**2)
    )


def cc_metrics(rollout: dict[str, np.ndarray], event_step: int = 120) -> dict[str, float]:
    """Compute path and heading metrics used in central-control experiments."""

    path_error = rollout["path_error"]
    heading = np.abs(
        np.arctan2(
            np.sin(rollout["target"][:, 2] - rollout["state"][:, 2]),
            np.cos(rollout["target"][:, 2] - rollout["state"][:, 2]),
        )
    )
    late = max(event_step + 20, len(path_error) - 80)
    return {
        "rms_path_error": float(np.sqrt(np.mean(path_error * path_error))),
        "late_path_error": float(np.mean(path_error[late:])),
        "late_heading_error": float(np.mean(heading[late:])),
        "post_event_path_integral": float(np.sum(path_error[event_step:]) * 0.05),
    }
