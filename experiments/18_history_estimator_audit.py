#!/usr/bin/env python3
"""Controlled audit for the history-based functional estimator.

The audit keeps the causal rollout protocol from experiment 17 but adds:
validation-based early stopping, parameter-matched context/message MLPs,
additional held-out route families, repeated force-noise realizations, and
held-out calibration diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_PATH = Path(__file__).with_name("17_history_functional_amortized.py")
SPEC = importlib.util.spec_from_file_location("history_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)


TRAIN_TASKS = ["straight", "left", "right", "s_lr", "s_rl"]
TEST_TASKS = ["speed", "sine", "chirp"]
MODEL_SEEDS = [41, 42, 43]
Z_VALUES = {"50": 0.67448975, "80": 1.28155157, "95": 1.95996398}


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
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


def parameter_count(input_dim: int, hidden: int) -> int:
    return input_dim * hidden + hidden + hidden * hidden + hidden + hidden + 1


def matched_hidden(input_dim: int, target_parameters: int) -> int:
    candidates = range(8, 129)
    return min(candidates, key=lambda width: abs(parameter_count(input_dim, width) - target_parameters))


def normalize(train_x, train_y):
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(train_y.mean())
    y_scale = float(train_y.std()) or 1.0
    return x_mean, x_scale, y_mean, y_scale


def prepare(x, y, normalization):
    x_mean, x_scale, y_mean, y_scale = normalization
    return (
        torch.as_tensor((x - x_mean) / x_scale, dtype=torch.float32),
        torch.as_tensor((y - y_mean) / y_scale, dtype=torch.float32),
    )


def score(model, normalization, x, y):
    x_tensor, y_tensor = prepare(x, y, normalization)
    model.eval()
    with torch.no_grad():
        prediction = model(x_tensor).numpy()
    residual = prediction - y_tensor.numpy()
    mse = float(np.mean(residual**2))
    coverage = {
        level: float(np.mean(np.abs(residual) <= z))
        for level, z in Z_VALUES.items()
    }
    return {
        "nll_nats": 0.5 * mse,
        "rmse_standardized": float(np.sqrt(mse)),
        "coverage_50": coverage["50"],
        "coverage_80": coverage["80"],
        "coverage_95": coverage["95"],
    }


def train_with_validation(train_x, train_y, val_x, val_y, seed, max_epochs, patience, hidden):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    normalization = normalize(train_x, train_y)
    x_train, y_train = prepare(train_x, train_y, normalization)
    x_val, y_val = prepare(val_x, val_y, normalization)
    model = MLP(x_train.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        prediction = model(x_train)
        loss = 0.5 * (prediction - y_train).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(0.5 * (model(x_val) - y_val).square().mean())
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, normalization, best_epoch


def collect_items(regime, seeds, task_names, condition_names, horizon, noise_replicates):
    items = {condition: [] for condition in condition_names}
    policies = BASE.load_policy_file("policies/four_architecture_policies.npz")
    for seed in seeds:
        controller = BASE.Controller.from_vector(regime, policies[(regime, seed)])
        for task_index, task_name in enumerate(task_names):
            task = BASE.make_task(task_name)
            for condition_index, condition in enumerate(condition_names):
                cc, fault, sensor_fault = BASE.CONDITIONS[condition]
                for replicate in range(noise_replicates):
                    sequence = BASE.collect_sequence(
                        controller,
                        task,
                        cc,
                        fault,
                        sensor_fault,
                        horizon,
                        seed + 100000 * replicate + 1000 * horizon + 100 * task_index + condition_index,
                    )
                    items[condition].append(sequence)
    return items


def window_pool(items, history):
    windows = [BASE.make_windows(*item, history) for item in items]
    return (
        np.vstack([part[0] for part in windows]),
        np.vstack([part[1] for part in windows]),
        np.concatenate([part[2] for part in windows]),
    )


def post_event_windows(items, history, event_step):
    context, message, target = window_pool(items, history)
    n_per_limb = target.size // 4
    keep = np.repeat(np.arange(history, history + n_per_limb) >= event_step, 4)
    return context[keep], message[keep], target[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/history_estimator_audit.csv")
    parser.add_argument("--histories", nargs="*", type=int, default=[0, 4, 8])
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 10, 40])
    parser.add_argument("--regimes", nargs="*", choices=BASE.REGIMES, default=["own_sensor", "peer_sensor"])
    parser.add_argument("--conditions", nargs="*", choices=list(BASE.CONDITIONS), default=list(BASE.CONDITIONS))
    parser.add_argument("--noise-replicates", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--context-hidden", type=int, default=64)
    args = parser.parse_args()

    policies = BASE.load_policy_file("policies/four_architecture_policies.npz")
    available_seeds = sorted(seed for regime, seed in policies if regime == args.regimes[0])
    train_seeds = available_seeds[:3]
    validation_seeds = available_seeds[3:4]
    test_seeds = available_seeds[4:]
    rows = []

    for history in args.histories:
        for horizon in args.horizons:
            for regime in args.regimes:
                train = collect_items(regime, train_seeds, TRAIN_TASKS, args.conditions, horizon, 2)
                validation = collect_items(regime, validation_seeds, TRAIN_TASKS, args.conditions, horizon, 2)
                test = collect_items(regime, test_seeds, TEST_TASKS, args.conditions, horizon, args.noise_replicates)
                train_context, train_message, train_y = window_pool(
                    [item for condition in args.conditions for item in train[condition]], history
                )
                val_context, val_message, val_y = window_pool(
                    [item for condition in args.conditions for item in validation[condition]], history
                )
                context_width = train_context.shape[1]
                message_width = train_message.shape[1]
                target_parameters = parameter_count(context_width, args.context_hidden)
                message_hidden = matched_hidden(context_width + message_width, target_parameters)
                for condition in args.conditions:
                    test_context, test_message, test_y = post_event_windows(
                        test[condition], history, BASE.EVENT_STEPS[condition]
                    )
                    test_context_x = test_context
                    test_message_x = np.column_stack((test_context, test_message))
                    val_context_x = val_context
                    val_message_x = np.column_stack((val_context, val_message))
                    for model_seed in MODEL_SEEDS:
                        context_model, context_norm, context_epoch = train_with_validation(
                            train_context,
                            train_y,
                            val_context_x,
                            val_y,
                            model_seed,
                            args.max_epochs,
                            args.patience,
                            args.context_hidden,
                        )
                        message_model, message_norm, message_epoch = train_with_validation(
                            np.column_stack((train_context, train_message)),
                            train_y,
                            val_message_x,
                            val_y,
                            model_seed,
                            args.max_epochs,
                            args.patience,
                            message_hidden,
                        )
                        models = [
                            ("context_only", context_model, context_norm, context_epoch, test_context_x),
                            ("with_message", message_model, message_norm, message_epoch, test_message_x),
                        ]
                        shuffled = test_message_x.copy()
                        permutation = np.random.default_rng(7000 + model_seed + history * 100 + horizon).permutation(len(shuffled))
                        shuffled[:, context_width:] = shuffled[permutation, context_width:]
                        models.append(("with_message_shuffled", message_model, message_norm, message_epoch, shuffled))
                        context_test = score(context_model, context_norm, test_context_x, test_y)
                        message_test = score(message_model, message_norm, test_message_x, test_y)
                        shuffled_test = score(message_model, message_norm, shuffled, test_y)
                        context_train = score(context_model, context_norm, train_context, train_y)
                        message_train = score(message_model, message_norm, np.column_stack((train_context, train_message)), train_y)
                        context_val = score(context_model, context_norm, val_context_x, val_y)
                        message_val = score(message_model, message_norm, val_message_x, val_y)
                        for name, current, train_score, val_score, test_score in [
                            ("context_only", context_model, context_train, context_val, context_test),
                            ("with_message", message_model, message_train, message_val, message_test),
                            ("with_message_shuffled", message_model, message_train, message_val, shuffled_test),
                        ]:
                            gain = ""
                            if name == "with_message":
                                gain = (context_test["nll_nats"] - message_test["nll_nats"]) / np.log(2.0)
                            rows.append({
                                "history": history,
                                "horizon": horizon,
                                "regime": regime,
                                "condition": condition,
                                "model_seed": model_seed,
                                "model": name,
                                "context_hidden": args.context_hidden,
                                "message_hidden": message_hidden,
                                "context_parameters": target_parameters,
                                "message_parameters": parameter_count(context_width + message_width, message_hidden),
                                "best_epoch": context_epoch if name == "context_only" else message_epoch,
                                "train_nll_nats": train_score["nll_nats"],
                                "validation_nll_nats": val_score["nll_nats"],
                                "test_nll_nats": test_score["nll_nats"],
                                "train_rmse_standardized": train_score["rmse_standardized"],
                                "validation_rmse_standardized": val_score["rmse_standardized"],
                                "test_rmse_standardized": test_score["rmse_standardized"],
                                "test_coverage_50": test_score["coverage_50"],
                                "test_coverage_80": test_score["coverage_80"],
                                "test_coverage_95": test_score["coverage_95"],
                                "gain_bits_per_transition": gain,
                            })
                        print(
                            f"k={history:2d} H={horizon:2d} {regime:12s} {condition:38s} "
                            f"gain={(context_test['nll_nats'] - message_test['nll_nats']) / np.log(2.0):.5f} "
                            f"epochs={context_epoch}/{message_epoch}"
                        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
