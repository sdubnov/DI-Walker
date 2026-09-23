#!/usr/bin/env python3
"""Create the current functional-results figure used by the compact paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = pd.read_csv(ROOT / "results" / "four_architecture_cc_dropout_summary.csv")
    data = data[data["regime"].isin(["own_sensor", "peer_sensor"])].copy()
    scenarios = [
        "CC dropout only",
        "L4 loss only",
        "L4 loss + CC dropout",
        "L4 slip only",
        "L4 slip + CC dropout",
    ]
    labels = ["CC dropout", "L4 loss", "L4 loss + CC", "L4 slip", "L4 slip + CC"]
    colors = {"own_sensor": "#2d78b5", "peer_sensor": "#d97706"}
    names = {"own_sensor": "Own-Sensor", "peer_sensor": "Peer-Sensor"}

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=220)
    x = range(len(scenarios))
    width = 0.34
    for offset, regime in [(-width / 2, "own_sensor"), (width / 2, "peer_sensor")]:
        subset = data.set_index(["scenario", "regime"]).loc[
            [(scenario, regime) for scenario in scenarios]
        ]
        means = subset["mean"].to_numpy()
        errors = subset["std"].to_numpy()
        positions = [i + offset for i in x]
        ax.bar(
            positions,
            means,
            width=width,
            yerr=errors,
            capsize=4,
            color=colors[regime],
            alpha=0.9,
            label=names[regime],
        )

    ax.set_title("Functional robustness under disturbance", fontsize=18, weight="bold")
    ax.set_ylabel("Mean late path error", fontsize=14)
    ax.set_xticks(list(x), labels, rotation=18, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "paper_functional_robustness.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
