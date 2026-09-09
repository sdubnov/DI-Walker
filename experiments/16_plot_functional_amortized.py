#!/usr/bin/env python3
"""Plot completed functional-amortized-model CSV files without rerunning simulations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.inputs)
    horizons = sorted({int(row["horizon"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    conditions = ["limb_slip_sensor_retained", "limb_loss_sensor_loss_cc_dropout"]
    for axis, regime in zip(axes, ["own_sensor", "peer_sensor"]):
        for condition in conditions:
            means, stds = [], []
            for horizon in horizons:
                values = [
                    float(row["gain_bits_per_transition"])
                    for row in rows
                    if row["regime"] == regime
                    and row["condition"] == condition
                    and int(row["horizon"]) == horizon
                    and row["model"] == "with_message"
                ]
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
            axis.errorbar(horizons, means, yerr=stds, marker="o", capsize=3, label=condition)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlabel("Prediction horizon H (steps)")
        axis.set_ylabel("Functional message gain (bits/transition)")
        axis.set_title(regime)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Embodied functional predictive gain versus horizon")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)


if __name__ == "__main__":
    main()
