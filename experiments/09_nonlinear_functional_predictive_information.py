#!/usr/bin/env python3
"""Estimate functional predictive information with nonlinear density models."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task
from robust_walker.config import FORCE_SCALE, DT


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
MESSAGE_WIDTH = {"no_sensor": 0, "own_sensor": 4, "peer_sensor": 12, "all_linear": 16}


class GaussianMLP(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        return self.mean(h).squeeze(-1), self.logvar(h).squeeze(-1).clamp(-4.0, 3.0)


def signed_cross_track(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    normal = np.column_stack((-np.sin(target[:, 2]), np.cos(target[:, 2])))
    return np.sum((target[:, :2] - state[:, :2]) * normal, axis=1)


def routed_messages(regime: str, sensed: np.ndarray) -> np.ndarray:
    if regime == "no_sensor":
        return np.empty((len(sensed), 0))
    if regime == "own_sensor":
        return sensed.copy()
    if regime == "peer_sensor":
        return np.concatenate([np.delete(sensed, i, axis=1) for i in range(4)], axis=1)
    if regime == "all_linear":
        return np.tile(sensed, (1, 4))
    raise ValueError(f"unknown regime: {regime}")


def collect_rollout(controller, task, cc, fault, horizon):
    result = rollout(controller, task, central_control=cc, limb_fault=fault)
    n = len(result["state"]) - horizon
    state = result["state"][:n]
    target = result["target"][:n]
    context = np.column_stack((
        state,
        result["act"][:n],
        task.vd[:n, None],
        task.wd[:n, None],
        result["error"][:n],
        result["cc"][:n],
    ))
    # Align messages with the values available before each controller update.
    messages = routed_messages(controller.regime.value, result["message"][:n] / FORCE_SCALE)
    features = np.column_stack((context, messages))
    target_error = signed_cross_track(result["state"][horizon:], result["target"][horizon:])
    return features, target_error


def fit_density(x: np.ndarray, y: np.ndarray, hidden: int, epochs: int, seed: int, validation=None):
    torch.manual_seed(seed)
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean, y_scale = float(y.mean()), max(float(y.std()), 1e-8)
    tx = torch.as_tensor((x - x_mean) / x_scale, dtype=torch.float32)
    ty = torch.as_tensor((y - y_mean) / y_scale, dtype=torch.float32)
    model = GaussianMLP(tx.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    model.train()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    stale = 0
    for _ in range(epochs):
        mean, logvar = model(tx)
        loss = 0.5 * (logvar + (ty - mean) ** 2 * torch.exp(-logvar)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                vx = torch.as_tensor((validation[0] - x_mean) / x_scale, dtype=torch.float32)
                vy = torch.as_tensor((validation[1] - y_mean) / y_scale, dtype=torch.float32)
                vm, vl = model(vx)
                validation_loss = float((0.5 * (vl + (vy - vm) ** 2 * torch.exp(-vl))).mean())
            model.train()
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= 35:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fitted_mean, _ = model(tx)
    residual_var = max(float(torch.mean((ty - fitted_mean) ** 2)) * y_scale * y_scale, 1e-6)
    return model, x_mean, x_scale, y_mean, y_scale, residual_var


def loglik(fitted, x: np.ndarray, y: np.ndarray, shared_variance: float | None = None) -> np.ndarray:
    model, x_mean, x_scale, y_mean, y_scale, fitted_variance = fitted
    with torch.no_grad():
        mean, logvar = model(torch.as_tensor((x - x_mean) / x_scale, dtype=torch.float32))
    mean = mean.numpy() * y_scale + y_mean
    variance = fitted_variance if shared_variance is None else shared_variance
    return -0.5 * (np.log(2.0 * np.pi * variance) + (y - mean) ** 2 / variance)


def estimate(rows, message_width, hidden, epochs, seed):
    if message_width == 0:
        return 0.0
    split = len(rows) // 2
    train_rows, test_rows = rows[:split], rows[split:]
    inner_split = max(2, len(train_rows) - 3)
    fit_rows, validation_rows = train_rows[:inner_split], train_rows[inner_split:]
    train_x = np.vstack([x for x, _ in fit_rows])
    train_y = np.concatenate([y for _, y in fit_rows])
    validation_x = np.vstack([x for x, _ in validation_rows])
    validation_y = np.concatenate([y for _, y in validation_rows])
    test_x = np.vstack([x for x, _ in test_rows])
    test_y = np.concatenate([y for _, y in test_rows])
    context_width = train_x.shape[1] - message_width
    base = fit_density(train_x[:, :context_width], train_y, hidden, epochs, seed, (validation_x[:, :context_width], validation_y))
    augmented = fit_density(train_x, train_y, hidden, epochs, seed + 10000, (validation_x, validation_y))
    # Use one variance for both models. This makes the comparison a stable
    # nonlinear mean-prediction test rather than a variance-fitting contest.
    return float(np.mean(loglik(augmented, test_x, test_y, base[5]) - loglik(base, test_x[:, :context_width], test_y, base[5])) / np.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/nonlinear_functional_predictive_information.csv")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=32)
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
                    collected.append(collect_rollout(controller, task, cc, fault, args.horizon))
            rate = estimate(collected, MESSAGE_WIDTH[regime], args.hidden, args.epochs, 7000 + len(rows))
            rows.append({"regime": regime, "condition": condition, "horizon": args.horizon, "RF_bits_per_sample": rate, "RF_bits_per_second": rate / DT})

    by_regime = {(r["regime"], r["condition"]): r["RF_bits_per_sample"] for r in rows}
    for row in rows:
        intact = by_regime[(row["regime"], "intact")]
        row["LF_bits_per_sample"] = intact - row["RF_bits_per_sample"]
        row["LF_bits_per_second"] = row["LF_bits_per_sample"] / DT

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
