#!/usr/bin/env python3
"""Global functional predictive-information pilot with embodied observations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import ForceNoise, LimbFault, SensorFault
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


REGIMES = ["no_sensor", "own_sensor", "peer_sensor", "all_linear"]
TASK_NAMES = ["straight", "s_lr", "s_rl", "sine", "chirp"]
CONDITIONS = {
    "intact": (CentralControl(), LimbFault(), SensorFault()),
    "limb_slip_sensor_retained": (CentralControl(), LimbFault(3, "slip", start_step=65), SensorFault()),
    "limb_loss_sensor_retained": (CentralControl(), LimbFault(3, "loss", strength=0.0, start_step=80), SensorFault()),
    "sensor_failure_only": (CentralControl(), LimbFault(), SensorFault(3, start_step=80)),
    "limb_loss_sensor_loss": (CentralControl(), LimbFault(3, "loss", strength=0.0, start_step=80), SensorFault(3, start_step=80)),
    "limb_loss_sensor_loss_cc_dropout": (
        CentralControl(0.10, "dropout", drop_step=120),
        LimbFault(3, "loss", strength=0.0, start_step=80),
        SensorFault(3, start_step=80),
    ),
}
EVENT_STEPS = {
    "intact": 0,
    "limb_slip_sensor_retained": 65,
    "limb_loss_sensor_retained": 80,
    "sensor_failure_only": 80,
    "limb_loss_sensor_loss": 80,
    "limb_loss_sensor_loss_cc_dropout": 80,
}
FORCE_SCALE = 1.18
DT = 0.05


def local_message(regime: str, sensed: np.ndarray, limb: int) -> np.ndarray:
    if regime == "no_sensor":
        return np.empty((len(sensed), 0))
    if regime == "own_sensor":
        return sensed[:, limb : limb + 1]
    if regime == "peer_sensor":
        return np.delete(sensed, limb, axis=1)
    if regime == "all_linear":
        return sensed.copy()
    raise ValueError(regime)


def signed_cross_track(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    normal = np.column_stack((-np.sin(target[:, 2]), np.cos(target[:, 2])))
    return np.sum((target[:, :2] - state[:, :2]) * normal, axis=1)


def collect_rollout(controller, task, cc, fault, sensor_fault, horizon, noise_sigma, noise_correlation, noise_seed, include_mask=False):
    result = rollout(
        controller,
        task,
        central_control=cc,
        limb_fault=fault,
        sensor_fault=sensor_fault,
        force_noise=ForceNoise(sigma=noise_sigma, correlation=noise_correlation, seed=noise_seed),
    )
    n = task.n_steps - horizon
    target = task.target_path()
    future_error = signed_cross_track(result["state"][horizon:], target[horizon:])
    state = result["state_before"][:n]
    activation = result["act_before"][:n]
    error = result["error"][:n]
    sensed = result["message"][:n] / FORCE_SCALE
    available = result["message_available"][:n]
    rows = []
    targets = []
    for limb in range(4):
        local_context = np.column_stack(
            (
                state[:, 2:6],
                activation[:, limb],
                task.vd[:n],
                task.wd[:n],
                error,
                result["cc"][:n],
            )
        )
        message = local_message(controller.regime.value, sensed, limb)
        if include_mask:
            message = np.column_stack((message, local_message(controller.regime.value, available, limb)))
        rows.append(np.column_stack((local_context, message)))
        targets.append(future_error)
    return np.vstack(rows), np.concatenate(targets)


class GaussianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def normalize(train_x, train_y):
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(train_y.mean())
    y_scale = float(train_y.std())
    if y_scale < 1e-8:
        y_scale = 1.0
    return x_mean, x_scale, y_mean, y_scale


def train_model(train_x, train_y, seed, epochs, hidden):
    torch.manual_seed(seed)
    x_mean, x_scale, y_mean, y_scale = normalize(train_x, train_y)
    x = torch.as_tensor((train_x - x_mean) / x_scale, dtype=torch.float32)
    y = torch.as_tensor((train_y - y_mean) / y_scale, dtype=torch.float32)
    model = GaussianMLP(x.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        prediction = model(x)
        loss = 0.5 * (prediction - y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, (x_mean, x_scale, y_mean, y_scale)


def score_model(model, normalization, test_x, test_y):
    x_mean, x_scale, y_mean, y_scale = normalization
    x = torch.as_tensor((test_x - x_mean) / x_scale, dtype=torch.float32)
    y = torch.as_tensor((test_y - y_mean) / y_scale, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        mse = (prediction - y).square().mean().item()
    # Fixed unit Gaussian in standardized target coordinates.
    return 0.5 * mse, float(np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/functional_amortized_model.csv")
    parser.add_argument("--plot", default="results/functional_amortized_horizon.png")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 10, 40])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-seeds", type=int, default=6)
    parser.add_argument("--seed-split", type=int, default=4)
    parser.add_argument("--noise-sigma", type=float, default=0.15)
    parser.add_argument("--noise-correlation", type=float, default=0.5)
    parser.add_argument("--include-availability-mask", action="store_true")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for horizon in args.horizons:
        for regime in REGIMES:
            selected = sorted(
                [(seed, params) for (name, seed), params in policies.items() if name == regime]
            )[: args.max_seeds]
            train_seeds = {seed for seed, _ in selected[: args.seed_split]}
            train_data = {condition: [] for condition in CONDITIONS}
            test_data = {condition: [] for condition in CONDITIONS}
            for seed, params in selected:
                controller = Controller.from_vector(regime, params)
                for task_index, task_name in enumerate(TASK_NAMES):
                    task = make_task(task_name)
                    if seed in train_seeds and task_index < 3:
                        destination = train_data
                    elif seed not in train_seeds and task_index >= 3:
                        destination = test_data
                    else:
                        continue
                    for condition_index, (condition, (cc, fault, sensor_fault)) in enumerate(CONDITIONS.items()):
                        x, y = collect_rollout(
                            controller,
                            task,
                            cc,
                            fault,
                            sensor_fault,
                            horizon,
                            args.noise_sigma,
                            args.noise_correlation,
                            seed + 1000 * horizon + 100 * task_index + condition_index,
                            args.include_availability_mask,
                        )
                        destination[condition].append((x, y))

            train_x_all = np.vstack([x for condition in CONDITIONS for x, _ in train_data[condition]])
            train_y_all = np.concatenate([y for condition in CONDITIONS for _, y in train_data[condition]])
            base_width = 4 + 1 + 2 + 3 + 1
            if args.include_availability_mask:
                message_width = {"no_sensor": 0, "own_sensor": 2, "peer_sensor": 6, "all_linear": 8}[regime]
            else:
                message_width = {"no_sensor": 0, "own_sensor": 1, "peer_sensor": 3, "all_linear": 4}[regime]
            context_width = base_width
            model_seeds = [41, 42, 43]
            for condition in CONDITIONS:
                test_parts = []
                target_parts = []
                for x, y in test_data[condition]:
                    n_steps = len(y) // 4
                    mask = np.arange(n_steps) >= EVENT_STEPS[condition]
                    mask = np.tile(mask, 4)
                    test_parts.append(x[mask])
                    target_parts.append(y[mask])
                test_x = np.vstack(test_parts)
                test_y = np.concatenate(target_parts)
                for model_seed in model_seeds:
                    context_model, context_norm = train_model(
                        train_x_all[:, :context_width], train_y_all,
                        model_seed, args.epochs, args.hidden,
                    )
                    message_model, message_norm = train_model(
                        train_x_all, train_y_all,
                        model_seed, args.epochs, args.hidden,
                    )
                    context_scores = score_model(context_model, context_norm, test_x[:, :context_width], test_y)
                    message_scores = score_model(message_model, message_norm, test_x, test_y)
                    shuffled_x = test_x.copy()
                    if message_width:
                        permutation = np.random.default_rng(7000 + model_seed + horizon).permutation(len(test_x))
                        shuffled_x[:, context_width:] = test_x[permutation, context_width:]
                    shuffled_scores = score_model(message_model, message_norm, shuffled_x, test_y)
                    gain = (context_scores[0] - message_scores[0]) / np.log(2.0)
                    rows.extend(
                        [
                            {
                                "regime": regime,
                                "condition": condition,
                                "horizon": horizon,
                                "model_seed": model_seed,
                                "model": "context_only",
                                "nll_nats": context_scores[0],
                                "rmse_standardized": context_scores[1],
                                "gain_bits_per_transition": "",
                            },
                            {
                                "regime": regime,
                                "condition": condition,
                                "horizon": horizon,
                                "model_seed": model_seed,
                                "model": "with_message",
                                "nll_nats": message_scores[0],
                                "rmse_standardized": message_scores[1],
                                "gain_bits_per_transition": gain,
                            },
                            {
                                "regime": regime,
                                "condition": condition,
                                "horizon": horizon,
                                "model_seed": model_seed,
                                "model": "with_message_shuffled",
                                "nll_nats": shuffled_scores[0],
                                "rmse_standardized": shuffled_scores[1],
                                "gain_bits_per_transition": "",
                            },
                        ]
                    )
                gains = [
                    float(row["gain_bits_per_transition"])
                    for row in rows
                    if row["regime"] == regime
                    and row["condition"] == condition
                    and row["horizon"] == horizon
                    and row["model"] == "with_message"
                ]
                print(f"H={horizon:2d} {regime:12s} {condition:38s} gain={np.mean(gains):.5f} +/- {np.std(gains):.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, regime in zip(axes, ["own_sensor", "peer_sensor"]):
        for condition in ["limb_slip_sensor_retained", "limb_loss_sensor_loss_cc_dropout"]:
            means = []
            stds = []
            for horizon in args.horizons:
                values = [
                    float(row["gain_bits_per_transition"])
                    for row in rows
                    if row["regime"] == regime
                    and row["condition"] == condition
                    and row["horizon"] == horizon
                    and row["model"] == "with_message"
                ]
                means.append(np.mean(values))
                stds.append(np.std(values))
            axis.errorbar(args.horizons, means, yerr=stds, marker="o", capsize=3, label=condition)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlabel("Prediction horizon H (steps)")
        axis.set_ylabel("Functional message gain (bits/transition)")
        axis.set_title(regime)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Embodied functional predictive gain versus horizon")
    fig.savefig(args.plot, dpi=220)


if __name__ == "__main__":
    main()
