#!/usr/bin/env python3
"""History-based embodied functional predictive-gain experiment.

This is the sequential extension of experiment 15.  It uses flattened causal
windows rather than an RNN, so the added predictive value of message history
can be audited without introducing recurrent-state implementation choices.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

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


def collect_sequence(controller, task, cc, fault, sensor_fault, horizon, noise_seed):
    result = rollout(
        controller,
        task,
        central_control=cc,
        limb_fault=fault,
        sensor_fault=sensor_fault,
        force_noise=ForceNoise(sigma=0.15, correlation=0.5, seed=noise_seed),
    )
    n = task.n_steps - horizon
    target = task.target_path()
    future_error = signed_cross_track(result["state"][horizon:], target[horizon:])
    state = result["state_before"][:n]
    activation = result["act_before"][:n]
    sensed = result["message"][:n] / FORCE_SCALE
    context_rows = []
    message_rows = []
    for limb in range(4):
        context = np.column_stack(
            (
                state[:, 2:6],
                activation[:, limb],
                task.vd[:n],
                task.wd[:n],
                result["error"][:n],
                result["cc"][:n],
            )
        )
        context_rows.append(context)
        message_rows.append(local_message(controller.regime.value, sensed, limb))
    return np.asarray(context_rows), np.asarray(message_rows), np.tile(future_error, (4, 1))


def make_windows(context, message, target, history):
    n_limbs, n_steps, _ = context.shape
    context_windows, message_windows, targets = [], [], []
    for limb in range(n_limbs):
        for t in range(history, n_steps):
            context_windows.append(context[limb, t - history : t + 1].reshape(-1))
            message_windows.append(message[limb, t - history : t + 1].reshape(-1))
            targets.append(target[limb, t])
    return np.asarray(context_windows), np.asarray(message_windows), np.asarray(targets)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def normalize(train_x, train_y):
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(train_y.mean())
    y_scale = float(train_y.std()) or 1.0
    return x_mean, x_scale, y_mean, y_scale


def train_model(train_x, train_y, seed, epochs, hidden):
    torch.manual_seed(seed)
    normalization = normalize(train_x, train_y)
    x_mean, x_scale, y_mean, y_scale = normalization
    x = torch.as_tensor((train_x - x_mean) / x_scale, dtype=torch.float32)
    y = torch.as_tensor((train_y - y_mean) / y_scale, dtype=torch.float32)
    model = MLP(x.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        prediction = model(x)
        loss = 0.5 * (prediction - y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, normalization


def score_model(model, normalization, test_x, test_y):
    x_mean, x_scale, y_mean, y_scale = normalization
    x = torch.as_tensor((test_x - x_mean) / x_scale, dtype=torch.float32)
    y = torch.as_tensor((test_y - y_mean) / y_scale, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        mse = (model(x) - y).square().mean().item()
    return 0.5 * mse, float(np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/history_functional_amortized.csv")
    parser.add_argument("--histories", nargs="*", type=int, default=[0, 4, 8])
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 10, 40])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-seeds", type=int, default=6)
    parser.add_argument("--seed-split", type=int, default=4)
    parser.add_argument("--regimes", nargs="*", choices=REGIMES, default=REGIMES)
    parser.add_argument("--conditions", nargs="*", choices=list(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for history in args.histories:
        for horizon in args.horizons:
            for regime in args.regimes:
                selected = sorted(
                    [(seed, params) for (name, seed), params in policies.items() if name == regime]
                )[: args.max_seeds]
                train_seeds = {seed for seed, _ in selected[: args.seed_split]}
                train_data = {condition: [] for condition in args.conditions}
                test_data = {condition: [] for condition in args.conditions}
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
                        for condition_index, condition in enumerate(args.conditions):
                            cc, fault, sensor_fault = CONDITIONS[condition]
                            sequence = collect_sequence(
                                controller,
                                task,
                                cc,
                                fault,
                                sensor_fault,
                                horizon,
                                seed + 1000 * horizon + 100 * task_index + condition_index,
                            )
                            destination[condition].append(sequence)

                context_width = 11 * (history + 1)
                message_width = {"no_sensor": 0, "own_sensor": 1, "peer_sensor": 3, "all_linear": 4}[regime] * (history + 1)

                for condition in args.conditions:
                    test_parts = [
                        make_windows(*item, history)
                        for item in test_data[condition]
                    ]
                    test_context = np.vstack([part[0] for part in test_parts])
                    test_message = np.vstack([part[1] for part in test_parts])
                    test_y = np.concatenate([part[2] for part in test_parts])
                    event_mask = np.repeat(
                        np.arange(history, history + test_y.size // 4) >= EVENT_STEPS[condition], 4
                    )
                    test_context = test_context[event_mask]
                    test_message = test_message[event_mask]
                    test_y = test_y[event_mask]

                    train_windows = [
                        make_windows(*item, history)
                        for condition_items in train_data.values()
                        for item in condition_items
                    ]
                    train_context_w = np.vstack([part[0] for part in train_windows])
                    train_message_w = np.vstack([part[1] for part in train_windows])
                    train_y_w = np.concatenate([part[2] for part in train_windows])
                    train_x_context = train_context_w
                    train_x_message = np.column_stack((train_context_w, train_message_w))
                    test_x_context = test_context
                    test_x_message = np.column_stack((test_context, test_message))

                    for model_seed in [41, 42, 43]:
                        context_model, context_norm = train_model(
                            train_x_context, train_y_w, model_seed, args.epochs, args.hidden
                        )
                        message_model, message_norm = train_model(
                            train_x_message, train_y_w, model_seed, args.epochs, args.hidden
                        )
                        context_score = score_model(context_model, context_norm, test_x_context, test_y)
                        message_score = score_model(message_model, message_norm, test_x_message, test_y)
                        shuffled_x = test_x_message.copy()
                        if message_width:
                            permutation = np.random.default_rng(
                                7000 + model_seed + history * 100 + horizon
                            ).permutation(len(test_x_message))
                            shuffled_x[:, context_width:] = test_x_message[permutation, context_width:]
                        shuffled_score = score_model(message_model, message_norm, shuffled_x, test_y)
                        gain = (context_score[0] - message_score[0]) / np.log(2.0)
                        rows.extend(
                            [
                                {"history": history, "horizon": horizon, "regime": regime, "condition": condition,
                                 "model_seed": model_seed, "model": "context_only", "nll_nats": context_score[0],
                                 "rmse_standardized": context_score[1], "gain_bits_per_transition": ""},
                                {"history": history, "horizon": horizon, "regime": regime, "condition": condition,
                                 "model_seed": model_seed, "model": "with_message", "nll_nats": message_score[0],
                                 "rmse_standardized": message_score[1], "gain_bits_per_transition": gain},
                                {"history": history, "horizon": horizon, "regime": regime, "condition": condition,
                                 "model_seed": model_seed, "model": "with_message_shuffled", "nll_nats": shuffled_score[0],
                                 "rmse_standardized": shuffled_score[1], "gain_bits_per_transition": ""},
                            ]
                        )
                    values = [
                        float(row["gain_bits_per_transition"])
                        for row in rows
                        if row["history"] == history and row["horizon"] == horizon
                        and row["regime"] == regime and row["condition"] == condition
                        and row["model"] == "with_message"
                    ]
                    print(
                        f"k={history:2d} H={horizon:2d} {regime:12s} {condition:38s} "
                        f"gain={np.mean(values):.5f} +/- {np.std(values):.5f}"
                    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
