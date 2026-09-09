#!/usr/bin/env python3
"""Stable calibration audit: mean MLP plus validation-fitted scalar variance."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

BASE_PATH = Path(__file__).with_name("18_history_estimator_audit.py")
SPEC = importlib.util.spec_from_file_location("audit_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

MODEL_SEEDS = [41, 42, 43]
Z_VALUES = {"50": 0.67448975, "80": 1.28155157, "95": 1.95996398}


def predictions(model, norm, x, y):
    x_tensor, y_tensor = BASE.prepare(x, y, norm)
    model.eval()
    with BASE.torch.no_grad():
        mean = model(x_tensor).numpy()
    return mean, y_tensor.numpy()


def calibrated_score(model, norm, calibration_variance, x, y):
    mean, target = predictions(model, norm, x, y)
    residual = mean - target
    variance = max(float(calibration_variance), 0.05)
    nll = 0.5 * np.mean(residual**2 / variance + np.log(variance))
    return {
        "nll_nats": float(nll),
        "rmse_standardized": float(np.sqrt(np.mean(residual**2))),
        "calibration_variance": variance,
        "coverage_50": float(np.mean(np.abs(residual) <= Z_VALUES["50"] * np.sqrt(variance))),
        "coverage_80": float(np.mean(np.abs(residual) <= Z_VALUES["80"] * np.sqrt(variance))),
        "coverage_95": float(np.mean(np.abs(residual) <= Z_VALUES["95"] * np.sqrt(variance))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/history_scalar_calibration.csv")
    parser.add_argument("--histories", nargs="*", type=int, default=[0, 4, 8])
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 40])
    parser.add_argument("--regimes", nargs="*", choices=BASE.BASE.REGIMES, default=["own_sensor", "peer_sensor"])
    parser.add_argument("--conditions", nargs="*", choices=list(BASE.BASE.CONDITIONS), default=["limb_slip_sensor_retained", "limb_loss_sensor_retained", "limb_loss_sensor_loss_cc_dropout"])
    parser.add_argument("--train-noise-replicates", type=int, default=5)
    parser.add_argument("--validation-noise-replicates", type=int, default=3)
    parser.add_argument("--test-noise-replicates", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--context-hidden", type=int, default=64)
    args = parser.parse_args()

    policies = BASE.BASE.load_policy_file("policies/four_architecture_policies.npz")
    available_seeds = sorted(seed for regime, seed in policies if regime == args.regimes[0])
    train_seeds, validation_seeds, test_seeds = available_seeds[:3], available_seeds[3:4], available_seeds[4:]
    rows = []
    for history in args.histories:
        for horizon in args.horizons:
            for regime in args.regimes:
                train = BASE.collect_items(regime, train_seeds, BASE.TRAIN_TASKS, args.conditions, horizon, args.train_noise_replicates)
                validation = BASE.collect_items(regime, validation_seeds, BASE.TRAIN_TASKS, args.conditions, horizon, args.validation_noise_replicates)
                test = BASE.collect_items(regime, test_seeds, BASE.TEST_TASKS, args.conditions, horizon, args.test_noise_replicates)
                train_context, train_message, train_y = BASE.window_pool([item for c in args.conditions for item in train[c]], history)
                val_context, val_message, val_y = BASE.window_pool([item for c in args.conditions for item in validation[c]], history)
                context_width = train_context.shape[1]
                message_width = train_message.shape[1]
                target_parameters = BASE.parameter_count(context_width, args.context_hidden)
                message_hidden = BASE.matched_hidden(context_width + message_width, target_parameters)
                train_message_x = np.column_stack((train_context, train_message))
                val_message_x = np.column_stack((val_context, val_message))
                for condition in args.conditions:
                    test_context, test_message, test_y = BASE.post_event_windows(test[condition], history, BASE.BASE.EVENT_STEPS[condition])
                    test_message_x = np.column_stack((test_context, test_message))
                    for model_seed in MODEL_SEEDS:
                        context_model, context_norm, context_epoch = BASE.train_with_validation(train_context, train_y, val_context, val_y, model_seed, args.max_epochs, args.patience, args.context_hidden)
                        message_model, message_norm, message_epoch = BASE.train_with_validation(train_message_x, train_y, val_message_x, val_y, model_seed, args.max_epochs, args.patience, message_hidden)
                        context_val_mean, context_val_target = predictions(context_model, context_norm, val_context, val_y)
                        message_val_mean, message_val_target = predictions(message_model, message_norm, val_message_x, val_y)
                        context_variance = max(float(np.mean((context_val_mean - context_val_target) ** 2)), 0.05)
                        message_variance = max(float(np.mean((message_val_mean - message_val_target) ** 2)), 0.05)
                        shuffled = test_message_x.copy()
                        permutation = np.random.default_rng(7000 + model_seed + history * 100 + horizon).permutation(len(shuffled))
                        shuffled[:, context_width:] = shuffled[permutation, context_width:]
                        context_train = calibrated_score(context_model, context_norm, context_variance, train_context, train_y)
                        context_val = calibrated_score(context_model, context_norm, context_variance, val_context, val_y)
                        context_test = calibrated_score(context_model, context_norm, context_variance, test_context, test_y)
                        message_train = calibrated_score(message_model, message_norm, message_variance, train_message_x, train_y)
                        message_val = calibrated_score(message_model, message_norm, message_variance, val_message_x, val_y)
                        message_test = calibrated_score(message_model, message_norm, message_variance, test_message_x, test_y)
                        shuffled_test = calibrated_score(message_model, message_norm, message_variance, shuffled, test_y)
                        for name, train_score, val_score, test_score, epoch in [
                            ("context_only", context_train, context_val, context_test, context_epoch),
                            ("with_message", message_train, message_val, message_test, message_epoch),
                            ("with_message_shuffled", message_train, message_val, shuffled_test, message_epoch),
                        ]:
                            gain = "" if name != "with_message" else (context_test["nll_nats"] - message_test["nll_nats"]) / np.log(2.0)
                            rows.append({"history": history, "horizon": horizon, "regime": regime, "condition": condition, "model_seed": model_seed, "model": name, "context_hidden": args.context_hidden, "message_hidden": message_hidden, "context_parameters": target_parameters, "message_parameters": BASE.parameter_count(context_width + message_width, message_hidden), "best_epoch": epoch, "train_nll_nats": train_score["nll_nats"], "validation_nll_nats": val_score["nll_nats"], "test_nll_nats": test_score["nll_nats"], "calibration_variance": test_score["calibration_variance"], "test_rmse_standardized": test_score["rmse_standardized"], "test_coverage_50": test_score["coverage_50"], "test_coverage_80": test_score["coverage_80"], "test_coverage_95": test_score["coverage_95"], "gain_bits_per_transition": gain})
                        print(f"k={history:2d} H={horizon:2d} {regime:12s} {condition:38s} gain={(context_test['nll_nats'] - message_test['nll_nats']) / np.log(2.0):.5f} epochs={context_epoch}/{message_epoch}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
