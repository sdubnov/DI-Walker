#!/usr/bin/env python3
"""Plot numerical results from the quantized-message sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"no_sensor": "#555555", "own_sensor": "#2a6fbb", "peer_sensor": "#d97706", "all_linear": "#16805c"}
LABELS = {"no_sensor": "Local", "own_sensor": "Own-Sensor", "peer_sensor": "Peer-Sensor", "all_linear": "All-Linear"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/quantized_message_sweep.csv")
    parser.add_argument("--output", default="results/quantized_message_rate_distortion.png")
    args = parser.parse_args()

    with Path(args.input).open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ["bits_per_value", "rate_bits_per_second", "late_path_error"]:
            row[key] = float(row[key])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for regime in LABELS:
        for scenario, ax in zip(["limb_slip", "limb_loss_cc_dropout"], axes):
            grouped = {}
            for row in rows:
                if row["regime"] == regime and row["scenario"] == scenario:
                    grouped.setdefault(row["bits_per_value"], []).append(row)
            points = []
            for bits, values in sorted(grouped.items()):
                points.append((values[0]["rate_bits_per_second"], np.mean([v["late_path_error"] for v in values]), bits))
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=LABELS[regime], color=COLORS[regime], linewidth=2)
            for rate, error, bits in points:
                ax.annotate(f"{int(bits)}b", (rate, error), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
            ax.set_xlabel("Directed message rate (bits/s)")
            ax.set_ylabel("Mean late path error")
            ax.grid(alpha=0.25)
            ax.set_title("L4 intermittent slip" if scenario == "limb_slip" else "L4 loss + CC dropout")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Quantized realized-force messages: frozen-policy rate-distortion sweep", fontsize=14)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    print(args.output)


if __name__ == "__main__":
    main()
