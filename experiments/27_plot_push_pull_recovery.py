#!/usr/bin/env python3
"""Plot disturbance-relative recovery-horizon diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    rows = list(csv.DictReader(open("results/adversarial_push_pull_summary.csv")))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    colors = {"own_sensor": "tab:blue", "peer_sensor": "tab:orange"}
    labels = {"own_sensor": "Own-Sensor", "peer_sensor": "Peer-Sensor"}
    for regime, color in colors.items():
        selected = sorted(
            [
                row for row in rows
                if row["regime"] == regime and abs(float(row["cc_gain"]) - 0.10) < 1e-12
            ],
            key=lambda row: float(row["amplitude"]),
        )
        x = [float(row["amplitude"]) for row in selected]
        axes[0].plot(
            x,
            [float(row["recovery_horizon_20pct_mean"]) for row in selected],
            "o-",
            color=color,
            label=labels[regime],
        )
        axes[1].plot(
            x,
            [float(row["recovery_rate_mean"]) for row in selected],
            "o-",
            color=color,
            label=labels[regime],
        )
        axes[2].plot(
            x,
            [float(row["counterfactual_horizon_20pct_mean"]) for row in selected],
            "o-",
            color=color,
            label=labels[regime],
        )
        axes[3].plot(
            x,
            [float(row["counterfactual_rate_mean"]) for row in selected],
            "o-",
            color=color,
            label=labels[regime],
        )
    axes[0].set_xlabel("push-pull torque amplitude")
    axes[0].set_ylabel(r"$T_{20}$ recovery horizon (steps)")
    axes[1].set_xlabel("push-pull torque amplitude")
    axes[1].set_ylabel(r"$\kappa_{\mathrm{rec}}$ (per step)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("push-pull torque amplitude")
    axes[2].set_ylabel(r"counterfactual $T_{20}$ (steps)")
    axes[3].set_xlabel("push-pull torque amplitude")
    axes[3].set_ylabel(r"counterfactual $\kappa_{\mathrm{dist}}$ (per step)")
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Absolute and counterfactual recovery at CC gain = 0.10")
    output = Path("results/adversarial_push_pull_recovery_horizon.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
