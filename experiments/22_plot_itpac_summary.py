#!/usr/bin/env python3
"""Create integrated functional/action/calibration plots from completed CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(rows, field):
    values = [float(row[field]) for row in rows]
    return float(np.mean(values)), float(np.std(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional", default="results/history_scalar_calibration.csv")
    parser.add_argument("--action", default="results/action_directed_information.csv")
    parser.add_argument("--out", default="results/itpac_integrated_summary.png")
    args = parser.parse_args()

    functional = read_rows(args.functional)
    action = read_rows(args.action)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel A: functional gain versus plant/task horizon H, at history k=8.
    axis = axes[0, 0]
    for regime, color in [("own_sensor", "tab:blue"), ("peer_sensor", "tab:orange")]:
        for condition, linestyle in [
            ("limb_slip_sensor_retained", "-"),
            ("limb_loss_sensor_loss_cc_dropout", "--"),
        ]:
            points = []
            for horizon in sorted({int(r["horizon"]) for r in functional}):
                subset = [
                    r for r in functional
                    if r["regime"] == regime
                    and r["condition"] == condition
                    and int(r["history"]) == 8
                    and int(r["horizon"]) == horizon
                    and r["model"] == "with_message"
                ]
                if subset:
                    points.append((horizon, *mean_std(subset, "gain_bits_per_transition")))
            if points:
                x, y, error = zip(*points)
                axis.errorbar(x, y, yerr=error, marker="o", capsize=3, color=color, linestyle=linestyle, label=f"{regime}: {condition}")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Global functional gain vs. task horizon")
    axis.set_xlabel("Plant prediction horizon H (steps)")
    axis.set_ylabel("Gain (bits/transition)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    # Panel B: action-directed gain versus history length k.
    axis = axes[0, 1]
    for regime, color in [("own_sensor", "tab:blue"), ("peer_sensor", "tab:orange")]:
        for condition, linestyle in [
            ("limb_slip_sensor_retained", "-"),
            ("limb_loss_sensor_retained", "--"),
            ("limb_loss_sensor_loss_cc_dropout", ":"),
        ]:
            points = []
            for history in sorted({int(r["history"]) for r in action}):
                subset = [
                    r for r in action
                    if r["regime"] == regime
                    and r["condition"] == condition
                    and int(r["history"]) == history
                    and r["model"] == "with_message"
                ]
                if subset:
                    points.append((history, *mean_std(subset, "gain_bits_per_action")))
            if points:
                x, y, error = zip(*points)
                axis.errorbar(x, y, yerr=error, marker="o", capsize=3, color=color, linestyle=linestyle, label=f"{regime}: {condition}")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Action-directed gain vs. message history")
    axis.set_xlabel("History length k (steps)")
    axis.set_ylabel("Gain (bits/action)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)

    # Panel C: true versus shuffled message NLL in the compound condition.
    axis = axes[1, 0]
    labels = ["Context", "True message", "Shuffled message"]
    x = np.arange(len(labels))
    width = 0.36
    for offset, regime, color in [(-width / 2, "own_sensor", "tab:blue"), (width / 2, "peer_sensor", "tab:orange")]:
        means, errors = [], []
        for model in ["context_only", "with_message", "with_message_shuffled"]:
            subset = [
                r for r in action
                if r["regime"] == regime
                and r["condition"] == "limb_loss_sensor_loss_cc_dropout"
                and int(r["history"]) == 8
                and r["model"] == model
            ]
            mean, std = mean_std(subset, "test_nll_nats")
            means.append(mean)
            errors.append(std)
        axis.bar(x + offset, means, width, yerr=errors, capsize=3, label=regime, color=color)
    axis.set_xticks(x, labels, rotation=15)
    axis.set_title("Action prediction: true vs. shuffled messages")
    axis.set_ylabel("Test NLL (nats)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    # Panel D: empirical coverage under nominal Gaussian intervals.
    axis = axes[1, 1]
    nominal = [50, 80, 95]
    x = np.arange(len(nominal))
    width = 0.36
    for offset, regime, color in [(-width / 2, "own_sensor", "tab:blue"), (width / 2, "peer_sensor", "tab:orange")]:
        coverage = []
        for level in [50, 80, 95]:
            subset = [
                r for r in action
                if r["regime"] == regime
                and r["condition"] == "limb_loss_sensor_loss_cc_dropout"
                and int(r["history"]) == 8
                and r["model"] == "with_message"
            ]
            coverage.append(np.mean([float(r[f"test_coverage_{level}"]) for r in subset]) * 100)
        axis.bar(x + offset, coverage, width, label=regime, color=color)
    axis.plot(x, nominal, "k--", marker="o", label="Nominal")
    axis.set_xticks(x, ["50%", "80%", "95%"])
    axis.set_ylim(0, 105)
    axis.set_title("Action-predictor calibration")
    axis.set_ylabel("Observed coverage (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    fig.suptitle("DI-Walker: functional and IT-PAC action-directed measurements", fontsize=14)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)


if __name__ == "__main__":
    main()
