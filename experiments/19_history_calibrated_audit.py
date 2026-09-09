#!/usr/bin/env python3
"""Calibrated, higher-data audit using the same flattened-history MLP family."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

BASE_PATH = Path(__file__).with_name("18_history_estimator_audit.py")
SPEC = importlib.util.spec_from_file_location("audit_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

MODEL_SEEDS = [41, 42, 43]
Z_VALUES = {"50": 0.67448975, "80": 1.28155157, "95": 1.95996398}


class GaussianMLP(nn.Module):
    """Same two-hidden-layer MLP, with mean and log-variance outputs."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        mean, logvar = self.head(self.body(x)).unbind(-1)
        return mean, torch.clamp(logvar, -6.0, 3.0)


def parameter_count(input_dim: int, hidden: int) -> int:
    return input_dim * hidden + hidden + hidden * hidden + hidden + 2 * hidden + 2


def matched_hidden(input_dim: int, target_parameters: int) -> int:
    return min(range(8, 129), key=lambda width: abs(parameter_count(input_dim, width) - target_parameters))


def normalize(train_x, train_y):
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(train_y.mean())
    y_scale = float(train_y.std()) or 1.0
    return x_mean, x_scale, y_mean, y_scale


def tensors(x, y, norm):
    x_mean, x_scale, y_mean, y_scale = norm
    return (
        torch.as_tensor((x - x_mean) / x_scale, dtype=torch.float32),
        torch.as_tensor((y - y_mean) / y_scale, dtype=torch.float32),
    )


def gaussian_nll(mean, logvar, y):
    return 0.5 * (torch.exp(-logvar) * (mean - y).square() + logvar).mean()


def train_model(train_x, train_y, val_x, val_y, seed, max_epochs, patience, hidden):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    norm = normalize(train_x, train_y)
    x_train, y_train = tensors(train_x, train_y, norm)
    x_val, y_val = tensors(val_x, val_y, norm)
    model = GaussianMLP(x_train.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss = gaussian_nll(*model(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(gaussian_nll(*model(x_val), y_val))
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
    return model, norm, best_epoch


def score(model, norm, x, y):
    x_tensor, y_tensor = tensors(x, y, norm)
    model.eval()
    with torch.no_grad():
        mean, logvar = model(x_tensor)
    residual = (mean - y_tensor).numpy()
    variance = torch.exp(logvar).numpy()
    return {
        "nll_nats": float(gaussian_nll(mean, logvar, y_tensor)),
        "rmse_standardized": float(np.sqrt(np.mean(residual**2))),
        "mean_logvar": float(np.mean(logvar.numpy())),
        "coverage_50": float(np.mean(np.abs(residual) <= Z_VALUES["50"] * np.sqrt(variance))),
        "coverage_80": float(np.mean(np.abs(residual) <= Z_VALUES["80"] * np.sqrt(variance))),
        "coverage_95": float(np.mean(np.abs(residual) <= Z_VALUES["95"] * np.sqrt(variance))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/history_calibrated_audit.csv")
    parser.add_argument("--histories", nargs="*", type=int, default=[0, 4, 8])
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 10, 40])
    parser.add_argument("--regimes", nargs="*", choices=BASE.BASE.REGIMES, default=["own_sensor", "peer_sensor"])
    parser.add_argument("--conditions", nargs="*", choices=list(BASE.BASE.CONDITIONS), default=list(BASE.BASE.CONDITIONS))
    parser.add_argument("--train-noise-replicates", type=int, default=5)
    parser.add_argument("--validation-noise-replicates", type=int, default=3)
    parser.add_argument("--test-noise-replicates", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--context-hidden", type=int, default=64)
    args = parser.parse_args()

    conditions = args.conditions
    policies = BASE.BASE.load_policy_file("policies/four_architecture_policies.npz")
    available_seeds = sorted(seed for regime, seed in policies if regime == args.regimes[0])
    train_seeds, validation_seeds, test_seeds = available_seeds[:3], available_seeds[3:4], available_seeds[4:]
    rows = []
    for history in args.histories:
        for horizon in args.horizons:
            for regime in args.regimes:
                train = BASE.collect_items(regime, train_seeds, BASE.TRAIN_TASKS, conditions, horizon, args.train_noise_replicates)
                validation = BASE.collect_items(regime, validation_seeds, BASE.TRAIN_TASKS, conditions, horizon, args.validation_noise_replicates)
                test = BASE.collect_items(regime, test_seeds, BASE.TEST_TASKS, conditions, horizon, args.test_noise_replicates)
                train_context, train_message, train_y = BASE.window_pool([item for c in conditions for item in train[c]], history)
                val_context, val_message, val_y = BASE.window_pool([item for c in conditions for item in validation[c]], history)
                context_width = train_context.shape[1]
                message_width = train_message.shape[1]
                target_parameters = parameter_count(context_width, args.context_hidden)
                message_hidden = matched_hidden(context_width + message_width, target_parameters)
                for condition in conditions:
                    test_context, test_message, test_y = BASE.post_event_windows(test[condition], history, BASE.BASE.EVENT_STEPS[condition])
                    train_message_x = np.column_stack((train_context, train_message))
                    val_message_x = np.column_stack((val_context, val_message))
                    test_message_x = np.column_stack((test_context, test_message))
                    for model_seed in MODEL_SEEDS:
                        context_model, context_norm, context_epoch = train_model(train_context, train_y, val_context, val_y, model_seed, args.max_epochs, args.patience, args.context_hidden)
                        message_model, message_norm, message_epoch = train_model(train_message_x, train_y, val_message_x, val_y, model_seed, args.max_epochs, args.patience, message_hidden)
                        shuffled = test_message_x.copy()
                        permutation = np.random.default_rng(7000 + model_seed + history * 100 + horizon).permutation(len(shuffled))
                        shuffled[:, context_width:] = shuffled[permutation, context_width:]
                        context_train, context_val, context_test = score(context_model, context_norm, train_context, train_y), score(context_model, context_norm, val_context, val_y), score(context_model, context_norm, test_context, test_y)
                        message_train, message_val, message_test = score(message_model, message_norm, train_message_x, train_y), score(message_model, message_norm, val_message_x, val_y), score(message_model, message_norm, test_message_x, test_y)
                        shuffled_test = score(message_model, message_norm, shuffled, test_y)
                        for name, train_score, val_score, test_score, epoch in [
                            ("context_only", context_train, context_val, context_test, context_epoch),
                            ("with_message", message_train, message_val, message_test, message_epoch),
                            ("with_message_shuffled", message_train, message_val, shuffled_test, message_epoch),
                        ]:
                            gain = "" if name != "with_message" else (context_test["nll_nats"] - message_test["nll_nats"]) / np.log(2.0)
                            rows.append({"history": history, "horizon": horizon, "regime": regime, "condition": condition, "model_seed": model_seed, "model": name, "context_hidden": args.context_hidden, "message_hidden": message_hidden, "context_parameters": target_parameters, "message_parameters": parameter_count(context_width + message_width, message_hidden), "best_epoch": epoch, "train_nll_nats": train_score["nll_nats"], "validation_nll_nats": val_score["nll_nats"], "test_nll_nats": test_score["nll_nats"], "test_rmse_standardized": test_score["rmse_standardized"], "test_mean_logvar": test_score["mean_logvar"], "test_coverage_50": test_score["coverage_50"], "test_coverage_80": test_score["coverage_80"], "test_coverage_95": test_score["coverage_95"], "gain_bits_per_transition": gain})
                        print(f"k={history:2d} H={horizon:2d} {regime:12s} {condition:38s} gain={((context_test['nll_nats'] - message_test['nll_nats']) / np.log(2.0)):.5f} epochs={context_epoch}/{message_epoch}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
