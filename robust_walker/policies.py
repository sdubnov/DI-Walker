"""Frozen policy file helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_policy_file(path: str | Path) -> dict[tuple[str, int], np.ndarray]:
    """Load a frozen policy archive keyed by (regime, seed)."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"missing policy file: {path}. Run experiments/01_train_policies.py to regenerate it."
        )
    data = np.load(path)
    policies: dict[tuple[str, int], np.ndarray] = {}
    for key in data.files:
        regime, seed = key.rsplit("_", 1)
        policies[(regime, int(seed))] = data[key]
    return policies


def save_policy_file(path: str | Path, policies: dict[tuple[str, int], np.ndarray]) -> None:
    """Save policies in the archive format consumed by experiment scripts."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{regime}_{seed}": p for (regime, seed), p in policies.items()})
