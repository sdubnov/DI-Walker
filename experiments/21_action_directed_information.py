#!/usr/bin/env python3
"""Estimate action-directed predictive information from existing plant rollouts.

The target is the receiving limb's post-policy command control[i,t].  The
message is available before that command is generated.  This is the action
interface analogue of the global functional estimator and is closer to the
IT-PAC information-to-go construction.
"""

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


def collect_action_sequence(controller, task, cc, fault, sensor_fault, noise_seed):
    result = BASE.BASE.rollout(
        controller,
        task,
        central_control=cc,
        limb_fault=fault,
        sensor_fault=sensor_fault,
        force_noise=BASE.BASE.ForceNoise(sigma=0.15, correlation=0.5, seed=noise_seed),
    )
    n = task.n_steps
    state = result["state_before"]
    activation = result["act_before"]
    sensed = result["message"] / BASE.BASE.FORCE_SCALE
    control = result["control"]
    previous_control = np.vstack((np.zeros((1, 4)), control[:-1]))
    contexts, messages, targets = [], [], []
    for limb in range(4):
        context = np.column_stack(
            (
                state[:, 2:6],
                activation[:, limb],
                previous_control[:, limb],
                task.vd,
                task.wd,
                result["error"],
                result["cc"],
            )
        )
        contexts.append(context)
        messages.append(BASE.BASE.local_message(controller.regime.value, sensed, limb))
        targets.append(control[:, limb])
    return np.asarray(contexts), np.asarray(messages), np.asarray(targets)


def make_windows(context, message, target, history):
    n_limbs, n_steps, _ = context.shape
    context_windows, message_windows, targets = [], [], []
    for limb in range(n_limbs):
        for t in range(history, n_steps):
            context_windows.append(context[limb, t - history : t + 1].reshape(-1))
            message_windows.append(message[limb, t - history : t + 1].reshape(-1))
            targets.append(target[limb, t])
    return np.asarray(context_windows), np.asarray(message_windows), np.asarray(targets)


def collect_items(regime, seeds, task_names, condition_names, noise_replicates):
    policies = BASE.BASE.load_policy_file("policies/four_architecture_policies.npz")
    items = {condition: [] for condition in condition_names}
    for seed in seeds:
        controller = BASE.BASE.Controller.from_vector(regime, policies[(regime, seed)])
        for task_index, task_name in enumerate(task_names):
            task = BASE.BASE.make_task(task_name)
            for condition_index, condition in enumerate(condition_names):
                cc, fault, sensor_fault = BASE.BASE.CONDITIONS[condition]
                for replicate in range(noise_replicates):
                    items[condition].append(
                        collect_action_sequence(
                            controller,
                            task,
                            cc,
                            fault,
                            sensor_fault,
                            seed + 100000 * replicate + 100 * task_index + condition_index,
                        )
                    )
    return items


def window_pool(items, history):
    windows = [make_windows(*item, history) for item in items]
    return (
        np.vstack([part[0] for part in windows]),
        np.vstack([part[1] for part in windows]),
        np.concatenate([part[2] for part in windows]),
    )


def post_event_windows(items, history, event_step):
    """Return windows whose target timestamp is after the event.

    ``make_windows`` emits rows in limb-major order for each rollout.  Build
    the timestamp mask with that same order and reset it for every item so
    event filtering cannot cross rollout or limb boundaries.
    """

    selected = []
    for item in items:
        context, message, target = make_windows(*item, history)
        n_limbs, n_steps, _ = item[0].shape
        timestamps = np.tile(np.arange(history, n_steps), n_limbs)
        keep = timestamps >= event_step
        selected.append((context[keep], message[keep], target[keep]))
    if not selected:
        raise ValueError("post_event_windows requires at least one rollout")
    return (
        np.vstack([part[0] for part in selected]),
        np.vstack([part[1] for part in selected]),
        np.concatenate([part[2] for part in selected]),
    )


def score(model, norm, variance, x, y):
    x_tensor, y_tensor = BASE.prepare(x, y, norm)
    model.eval()
    with BASE.torch.no_grad():
        prediction = model(x_tensor).numpy()
    residual = prediction - y_tensor.numpy()
    variance = max(float(variance), 0.05)
    return {
        "nll_nats": float(0.5 * np.mean(residual**2 / variance + np.log(variance))),
        "rmse_standardized": float(np.sqrt(np.mean(residual**2))),
        "coverage_50": float(np.mean(np.abs(residual) <= Z_VALUES["50"] * np.sqrt(variance))),
        "coverage_80": float(np.mean(np.abs(residual) <= Z_VALUES["80"] * np.sqrt(variance))),
        "coverage_95": float(np.mean(np.abs(residual) <= Z_VALUES["95"] * np.sqrt(variance))),
        "calibration_variance": variance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/action_directed_information.csv")
    parser.add_argument("--histories", nargs="*", type=int, default=[0, 4, 8])
    parser.add_argument("--regimes", nargs="*", choices=BASE.BASE.REGIMES, default=["own_sensor", "peer_sensor"])
    parser.add_argument("--conditions", nargs="*", choices=list(BASE.BASE.CONDITIONS), default=list(BASE.BASE.CONDITIONS))
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
        for regime in args.regimes:
            train = collect_items(regime, train_seeds, BASE.TRAIN_TASKS, args.conditions, args.train_noise_replicates)
            validation = collect_items(regime, validation_seeds, BASE.TRAIN_TASKS, args.conditions, args.validation_noise_replicates)
            test = collect_items(regime, test_seeds, BASE.TEST_TASKS, args.conditions, args.test_noise_replicates)
            train_context, train_message, train_y = window_pool([item for c in args.conditions for item in train[c]], history)
            val_context, val_message, val_y = window_pool([item for c in args.conditions for item in validation[c]], history)
            context_width = train_context.shape[1]
            message_width = train_message.shape[1]
            target_parameters = BASE.parameter_count(context_width, args.context_hidden)
            message_hidden = BASE.matched_hidden(context_width + message_width, target_parameters)
            train_message_x = np.column_stack((train_context, train_message))
            val_message_x = np.column_stack((val_context, val_message))
            for condition in args.conditions:
                test_context, test_message, test_y = post_event_windows(test[condition], history, BASE.BASE.EVENT_STEPS[condition])
                test_message_x = np.column_stack((test_context, test_message))
                for model_seed in MODEL_SEEDS:
                    context_model, context_norm, context_epoch = BASE.train_with_validation(train_context, train_y, val_context, val_y, model_seed, args.max_epochs, args.patience, args.context_hidden)
                    message_model, message_norm, message_epoch = BASE.train_with_validation(train_message_x, train_y, val_message_x, val_y, model_seed, args.max_epochs, args.patience, message_hidden)
                    x_val, y_val = BASE.prepare(val_context, val_y, context_norm)
                    with BASE.torch.no_grad():
                        context_val_prediction = context_model(x_val).numpy()
                        context_val_target = y_val.numpy()
                    message_x_val, message_y_val = BASE.prepare(val_message_x, val_y, message_norm)
                    with BASE.torch.no_grad():
                        message_val_prediction = message_model(message_x_val).numpy()
                        message_y_val = message_y_val.numpy()
                    context_variance = max(float(np.mean((context_val_prediction - context_val_target) ** 2)), 0.05)
                    message_variance = max(float(np.mean((message_val_prediction - message_y_val) ** 2)), 0.05)
                    shuffled = test_message_x.copy()
                    permutation = np.random.default_rng(7000 + model_seed + history * 100).permutation(len(shuffled))
                    shuffled[:, context_width:] = shuffled[permutation, context_width:]
                    context_train = score(context_model, context_norm, context_variance, train_context, train_y)
                    context_val = score(context_model, context_norm, context_variance, val_context, val_y)
                    context_test = score(context_model, context_norm, context_variance, test_context, test_y)
                    message_train = score(message_model, message_norm, message_variance, train_message_x, train_y)
                    message_val = score(message_model, message_norm, message_variance, val_message_x, val_y)
                    message_test = score(message_model, message_norm, message_variance, test_message_x, test_y)
                    shuffled_test = score(message_model, message_norm, message_variance, shuffled, test_y)
                    for name, train_score, val_score, test_score, epoch in [
                        ("context_only", context_train, context_val, context_test, context_epoch),
                        ("with_message", message_train, message_val, message_test, message_epoch),
                        ("with_message_shuffled", message_train, message_val, shuffled_test, message_epoch),
                    ]:
                        gain = "" if name != "with_message" else (context_test["nll_nats"] - message_test["nll_nats"]) / np.log(2.0)
                        rows.append({"history": history, "regime": regime, "condition": condition, "model_seed": model_seed, "model": name, "context_hidden": args.context_hidden, "message_hidden": message_hidden, "context_parameters": target_parameters, "message_parameters": BASE.parameter_count(context_width + message_width, message_hidden), "best_epoch": epoch, "train_nll_nats": train_score["nll_nats"], "validation_nll_nats": val_score["nll_nats"], "test_nll_nats": test_score["nll_nats"], "calibration_variance": test_score["calibration_variance"], "test_rmse_standardized": test_score["rmse_standardized"], "test_coverage_50": test_score["coverage_50"], "test_coverage_80": test_score["coverage_80"], "test_coverage_95": test_score["coverage_95"], "gain_bits_per_action": gain})
                    print(f"k={history:2d} {regime:12s} {condition:38s} gain={(context_test['nll_nats'] - message_test['nll_nats']) / np.log(2.0):.5f} epochs={context_epoch}/{message_epoch}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
