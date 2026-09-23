#!/usr/bin/env python3
"""Predictor-only message quantization for IT-PAC-style action information.

The plant is rolled out once with full-precision messages. Quantization is
applied only to the logged causal message before fitting the action predictors;
the action targets and plant trajectories are unchanged across levels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = Path(__file__).with_name("21_action_directed_information.py")
SPEC = importlib.util.spec_from_file_location("action_base", BASE_PATH)
ACTION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ACTION)

MODEL_SEEDS = [41, 42, 43]
LEVELS = [1, 3, 5, 9, 17, 33, 65]
REGIMES = ["own_sensor", "peer_sensor"]
CONDITIONS = ["intact", "limb_loss_sensor_loss_cc_dropout"]
CHANNELS_PER_RECEIVER = {"own_sensor": 1, "peer_sensor": 3}


def quantize_message(message: np.ndarray, levels: int | None) -> np.ndarray:
    """Quantize normalized message values with a fixed symmetric alphabet."""

    if levels is None:
        return message.copy()
    if levels < 1 or levels % 2 == 0:
        raise ValueError("levels must be a positive odd integer")
    if levels == 1:
        return np.zeros_like(message)
    clipped = np.clip(message, -1.0, 1.0)
    step = 2.0 / (levels - 1)
    return step * np.round(clipped / step)


def quantize_items(items, levels):
    return [(context, quantize_message(message, levels), target) for context, message, target in items]


def windows(items, history, post_event=None):
    if post_event is None:
        return ACTION.window_pool(items, history)
    return ACTION.post_event_windows(items, history, post_event)


def calibrated_score(model, norm, variance, x, y):
    return ACTION.score(model, norm, variance, x, y)


def plot_rows(rows, histories, output):
    """Plot seed means with standard-deviation bars, not seed-order joins."""

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    colors = {
        ("own_sensor", "intact"): "tab:blue",
        ("own_sensor", "limb_loss_sensor_loss_cc_dropout"): "tab:purple",
        ("peer_sensor", "intact"): "tab:orange",
        ("peer_sensor", "limb_loss_sensor_loss_cc_dropout"): "tab:red",
    }
    labels = {
        "intact": "intact",
        "limb_loss_sensor_loss_cc_dropout": "compound failure",
    }
    for regime in REGIMES:
        for condition, linestyle in [
            ("intact", "-"),
            ("limb_loss_sensor_loss_cc_dropout", "--"),
        ]:
            subset = [
                row for row in rows
                if row["regime"] == regime
                and row["condition"] == condition
                and int(row["history"]) == max(histories)
                and row["levels"] != "full_precision"
                and float(row["rate_per_receiver_bits_per_step"]) > 0
            ]
            rates = sorted({float(row["rate_per_receiver_bits_per_step"]) for row in subset})
            if not rates:
                continue
            values = []
            shuffled = []
            efficiencies = []
            for rate in rates:
                at_rate = [
                    row for row in subset
                    if float(row["rate_per_receiver_bits_per_step"]) == rate
                ]
                values.append([float(row["gain_bits_per_action"]) for row in at_rate])
                shuffled.append([float(row["shuffle_gap_bits_per_action"]) for row in at_rate])
                efficiencies.append([float(row["eta_per_receiver"]) for row in at_rate])

            means = [np.mean(value) for value in values]
            stds = [np.std(value) for value in values]
            shuffle_means = [np.mean(value) for value in shuffled]
            shuffle_stds = [np.std(value) for value in shuffled]
            efficiency_means = [np.mean(value) for value in efficiencies]
            efficiency_stds = [np.std(value) for value in efficiencies]
            label = f"{regime.replace('_', '-')}, {labels[condition]}"
            style = dict(
                color=colors[(regime, condition)],
                linestyle=linestyle,
                marker="o",
                capsize=3,
                label=label,
            )
            axes[0].errorbar(rates, means, yerr=stds, **style)
            axes[1].errorbar(rates, shuffle_means, yerr=shuffle_stds, **style)
            axes[2].errorbar(rates, efficiency_means, yerr=efficiency_stds, **style)

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Action-predictive gain")
    axes[1].set_title("True-message advantage")
    axes[2].set_title("Descriptive action efficiency")
    axes[0].set_ylabel("bits/action")
    axes[1].set_ylabel("bits/action")
    axes[2].set_ylabel("predictive bits / nominal bit")
    for axis in axes:
        axis.set_xlabel("Per-receiver nominal rate (bits/step)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Predictor-only message quantization: IT-PAC-style action information")
    figure.savefig(output, dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/predictor_quantization_itpac.csv")
    parser.add_argument("--plot", default="results/predictor_quantization_itpac.png")
    parser.add_argument("--histories", nargs="*", type=int, default=[0, 4, 8])
    parser.add_argument("--levels", nargs="*", type=int, default=LEVELS)
    parser.add_argument("--train-noise-replicates", type=int, default=5)
    parser.add_argument("--validation-noise-replicates", type=int, default=3)
    parser.add_argument("--test-noise-replicates", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--context-hidden", type=int, default=64)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        with Path(args.out).open() as handle:
            plot_rows(list(csv.DictReader(handle)), args.histories, args.plot)
        print(f"Plot: {args.plot}")
        return

    policies = ACTION.BASE.BASE.load_policy_file("policies/four_architecture_policies.npz")
    rows_by_regime = {}
    for regime in REGIMES:
        available_seeds = sorted(seed for policy_regime, seed in policies if policy_regime == regime)
        rows_by_regime[regime] = (
            available_seeds[:3],
            available_seeds[3:4],
            available_seeds[4:],
        )

    rows = []
    for regime in REGIMES:
        train_seeds, validation_seeds, test_seeds = rows_by_regime[regime]
        train = ACTION.collect_items(
            regime,
            train_seeds,
            ACTION.BASE.TRAIN_TASKS,
            CONDITIONS,
            args.train_noise_replicates,
        )
        validation = ACTION.collect_items(
            regime,
            validation_seeds,
            ACTION.BASE.TRAIN_TASKS,
            CONDITIONS,
            args.validation_noise_replicates,
        )
        test = ACTION.collect_items(
            regime,
            test_seeds,
            ACTION.BASE.TEST_TASKS,
            CONDITIONS,
            args.test_noise_replicates,
        )

        for history in args.histories:
            train_context, _, train_y = windows(
                [item for condition in CONDITIONS for item in train[condition]], history
            )
            val_context, _, val_y = windows(
                [item for condition in CONDITIONS for item in validation[condition]], history
            )
            context_width = train_context.shape[1]
            target_parameters = ACTION.BASE.parameter_count(context_width, args.context_hidden)

            for model_seed in MODEL_SEEDS:
                context_model, context_norm, context_epoch = ACTION.BASE.train_with_validation(
                    train_context,
                    train_y,
                    val_context,
                    val_y,
                    model_seed,
                    args.max_epochs,
                    args.patience,
                    args.context_hidden,
                )
                context_val_score = calibrated_score(
                    context_model,
                    context_norm,
                    0.05,
                    val_context,
                    val_y,
                )
                x_val, y_val = ACTION.BASE.prepare(val_context, val_y, context_norm)
                with ACTION.BASE.torch.no_grad():
                    context_val_prediction = context_model(x_val).numpy()
                    context_val_target = y_val.numpy()
                context_variance = max(
                    float(np.mean((context_val_prediction - context_val_target) ** 2)),
                    0.05,
                )

                for levels in [None, *args.levels]:
                    quantized_train = quantize_items(
                        [item for condition in CONDITIONS for item in train[condition]], levels
                    )
                    quantized_validation = quantize_items(
                        [item for condition in CONDITIONS for item in validation[condition]], levels
                    )
                    train_context_q, train_message_q, train_y_q = windows(quantized_train, history)
                    val_context_q, val_message_q, val_y_q = windows(quantized_validation, history)
                    train_message_x = np.column_stack((train_context_q, train_message_q))
                    val_message_x = np.column_stack((val_context_q, val_message_q))
                    message_width = train_message_q.shape[1]
                    message_hidden = ACTION.BASE.matched_hidden(
                        context_width + message_width,
                        target_parameters,
                    )
                    message_model, message_norm, message_epoch = ACTION.BASE.train_with_validation(
                        train_message_x,
                        train_y_q,
                        val_message_x,
                        val_y_q,
                        model_seed,
                        args.max_epochs,
                        args.patience,
                        message_hidden,
                    )
                    message_x_val, message_y_val = ACTION.BASE.prepare(val_message_x, val_y_q, message_norm)
                    with ACTION.BASE.torch.no_grad():
                        message_val_prediction = message_model(message_x_val).numpy()
                        message_val_target = message_y_val.numpy()
                    message_variance = max(
                        float(np.mean((message_val_prediction - message_val_target) ** 2)),
                        0.05,
                    )

                    for condition in CONDITIONS:
                        test_items = quantize_items(test[condition], levels)
                        test_context, test_message, test_y = windows(
                            test_items,
                            history,
                            ACTION.BASE.BASE.EVENT_STEPS[condition],
                        )
                        test_message_x = np.column_stack((test_context, test_message))
                        shuffled = test_message_x.copy()
                        context_width_test = test_context.shape[1]
                        permutation = np.random.default_rng(
                            7000 + model_seed + history * 100 + (levels or 0)
                        ).permutation(len(shuffled))
                        shuffled[:, context_width_test:] = shuffled[permutation, context_width_test:]

                        context_test = calibrated_score(
                            context_model,
                            context_norm,
                            context_variance,
                            test_context,
                            test_y,
                        )
                        message_test = calibrated_score(
                            message_model,
                            message_norm,
                            message_variance,
                            test_message_x,
                            test_y,
                        )
                        shuffled_test = calibrated_score(
                            message_model,
                            message_norm,
                            message_variance,
                            shuffled,
                            test_y,
                        )
                        # K=1 carries no message information.  Use the exact
                        # context-only score rather than a reparameterized
                        # zero-message MLP, so the zero-rate control is strict.
                        if levels == 1:
                            message_test = context_test
                            shuffled_test = context_test
                        gain = (context_test["nll_nats"] - message_test["nll_nats"]) / np.log(2.0)
                        shuffle_gap = (shuffled_test["nll_nats"] - message_test["nll_nats"]) / np.log(2.0)
                        rate_per_receiver = "" if levels is None else CHANNELS_PER_RECEIVER[regime] * np.log2(levels)
                        rate_shared_bus = "" if levels is None else 4.0 * np.log2(levels)
                        eta_receiver = "" if levels is None or rate_per_receiver == 0 else gain / rate_per_receiver
                        eta_shared = "" if levels is None or rate_shared_bus == 0 else 4.0 * gain / rate_shared_bus
                        rows.append(
                            {
                                "regime": regime,
                                "condition": condition,
                                "history": history,
                                "levels": "full_precision" if levels is None else levels,
                                "rate_per_receiver_bits_per_step": rate_per_receiver,
                                "rate_shared_bus_bits_per_step": rate_shared_bus,
                                "gain_bits_per_action": gain,
                                "shuffle_gap_bits_per_action": shuffle_gap,
                                "eta_per_receiver": eta_receiver,
                                "eta_shared_bus": eta_shared,
                                "context_test_nll_nats": context_test["nll_nats"],
                                "message_test_nll_nats": message_test["nll_nats"],
                                "shuffled_test_nll_nats": shuffled_test["nll_nats"],
                                "context_best_epoch": context_epoch,
                                "message_best_epoch": message_epoch,
                                "model_seed": model_seed,
                            }
                        )
                    print(
                        f"{regime:11s} k={history:2d} "
                        f"levels={levels or 'full':>4} seed={model_seed}"
                    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_rows(rows, args.histories, args.plot)
    print(f"CSV: {output}")
    print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
