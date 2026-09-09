#!/usr/bin/env python3
"""Estimate global functional predictive-information rates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.config import FORCE_SCALE
from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.faults import ForceNoise
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


REGIMES = ["no_sensor", "own_sensor", "peer_sensor", "all_linear"]
TASK_NAMES = ["straight", "s_lr", "s_rl", "sine", "chirp"]
CONDITIONS = {
    "intact": (CentralControl(), LimbFault()),
    "limb_loss": (CentralControl(), LimbFault(3, "loss", strength=0.0, start_step=80)),
    "limb_slip": (CentralControl(), LimbFault(3, "slip", start_step=65)),
    "limb_loss_cc_dropout": (
        CentralControl(0.10, "dropout", drop_step=120),
        LimbFault(3, "loss", strength=0.0, start_step=80),
    ),
}


def signed_cross_track(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return signed target-path cross-track error for each time step."""

    normal = np.column_stack((-np.sin(target[:, 2]), np.cos(target[:, 2])))
    return np.sum((target[:, :2] - state[:, :2]) * normal, axis=1)


def routed_messages(regime: str, sensed: np.ndarray) -> np.ndarray:
    """Represent the force-sensor values routed to the limb controllers."""

    if regime == "no_sensor":
        return np.empty((len(sensed), 0))
    if regime == "own_sensor":
        return sensed.copy()
    if regime == "peer_sensor":
        return np.concatenate([np.delete(sensed, i, axis=1) for i in range(4)], axis=1)
    if regime == "all_linear":
        return np.tile(sensed, (1, 4))
    raise ValueError(f"unknown regime: {regime}")


def collect_rollout(controller: Controller, task, cc, fault, horizon: int, noise_sigma: float, noise_correlation: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    result = rollout(
        controller,
        task,
        central_control=cc,
        limb_fault=fault,
        force_noise=ForceNoise(sigma=noise_sigma, correlation=noise_correlation, seed=seed),
    )
    n = len(result["state"]) - horizon
    if n <= 20:
        raise ValueError("rollout is shorter than the requested horizon")
    state = result["state"][:n]
    target = result["target"][:n]
    current_error = result["error"][:n]
    context = np.column_stack(
        (
            state,
            result["act"][:n],
            np.repeat(task.vd[:n, None], 1, axis=1),
            np.repeat(task.wd[:n, None], 1, axis=1),
            current_error,
            result["cc"][:n],
        )
    )
    # The controller reads the previous realized force before computing the
    # current action. Use the explicitly logged pre-action sensor message.
    message = routed_messages(controller.regime.value, result["message"][:n] / FORCE_SCALE)
    future_error = signed_cross_track(result["state"][horizon:], result["target"][horizon:])
    return np.column_stack((context, message)), future_error


def fit_gaussian_predictor(x_train: np.ndarray, y_train: np.ndarray, ridge: float = 1e-3):
    """Fit a standardized ridge mean model with a Gaussian residual."""

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x_train - mean) / scale
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    residual = y_train - design @ beta
    variance = max(float(np.mean(residual * residual)), 1e-8)
    return mean, scale, beta, variance


def gaussian_loglik(model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mean, scale, beta, variance = model
    z = (x - mean) / scale
    prediction = np.column_stack((np.ones(len(z)), z)) @ beta
    return -0.5 * (np.log(2.0 * np.pi * variance) + (y - prediction) ** 2 / variance)


def estimate_rate(rows: list[tuple[np.ndarray, np.ndarray]], message_width: int) -> float:
    """Estimate held-out message information in bits per sample."""

    if message_width == 0:
        return 0.0
    data = np.vstack([x for x, _ in rows])
    y = np.concatenate([target for _, target in rows])
    context_width = data.shape[1] - message_width
    split = max(1, len(rows) // 2)
    train = np.concatenate([np.arange(len(rows))[:split]], axis=0)
    # Split by rollout, not by adjacent time samples, to avoid temporal leakage.
    offsets = np.cumsum([0] + [len(target) for _, target in rows])
    train_ids = np.concatenate([np.arange(offsets[k], offsets[k + 1]) for k in train])
    test_ids = np.concatenate([np.arange(offsets[k], offsets[k + 1]) for k in range(split, len(rows))])
    x0_train, x1_train = data[train_ids, :context_width], data[train_ids]
    x0_test, x1_test = data[test_ids, :context_width], data[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]
    base = fit_gaussian_predictor(x0_train, y_train)
    augmented = fit_gaussian_predictor(x1_train, y_train)
    return float(np.mean(gaussian_loglik(augmented, x1_test, y_test) - gaussian_loglik(base, x0_test, y_test)) / np.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/functional_predictive_information.csv")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--noise-correlation", type=float, default=0.0)
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    tasks = [make_task(name) for name in TASK_NAMES]
    rows = []
    for regime in REGIMES:
        seed_policies = [(seed, p) for (name, seed), p in policies.items() if name == regime]
        if not seed_policies:
            continue
        for condition, (cc, fault) in CONDITIONS.items():
            collected = []
            for seed, params in seed_policies:
                controller = Controller.from_vector(regime, params)
                for task in tasks:
                    collected.append(collect_rollout(controller, task, cc, fault, args.horizon, args.noise_sigma, args.noise_correlation, seed + len(collected)))
            width = collected[0][0].shape[1] - (0 if regime == "no_sensor" else {"own_sensor": 4, "peer_sensor": 12, "all_linear": 16}[regime])
            message_width = collected[0][0].shape[1] - width
            rate = estimate_rate(collected, message_width)
            rows.append({"regime": regime, "condition": condition, "horizon": args.horizon, "RF_bits_per_sample": rate, "RF_bits_per_second": rate / 0.05})

    by_regime = {(row["regime"], row["condition"]): row["RF_bits_per_sample"] for row in rows}
    for row in rows:
        intact = by_regime[(row["regime"], "intact")]
        row["LF_bits_per_sample"] = intact - row["RF_bits_per_sample"]
        row["LF_bits_per_second"] = row["LF_bits_per_sample"] / 0.05

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['regime']:12s} {row['condition']:24s} RF={row['RF_bits_per_second']:.4f} bits/s LF={row['LF_bits_per_second']:.4f} bits/s")


if __name__ == "__main__":
    main()
