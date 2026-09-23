#!/usr/bin/env python3
"""Generate print-friendly monochrome figures for the compact paper."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRAY = {"own_sensor": "#222222", "peer_sensor": "#777777"}
REGIME_LABEL = {"own_sensor": "Own-Sensor", "peer_sensor": "Peer-Sensor"}


def system_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=220)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    boxes = [
        (0.4, 3.0, 2.4, 1.25, "realized peer force\n$F_{-i,t-1}$"),
        (3.35, 3.0, 2.4, 1.25, "quantized message\n$M_{i,t}$"),
        (6.3, 3.0, 2.4, 1.25, "receiving action\n$U_{i,t}$"),
        (9.25, 3.0, 2.4, 1.25, "next plant state\n$Z_{t+1}$"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#eeeeee", edgecolor="#222222", linewidth=1.5,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=15)

    ax.add_patch(patches.FancyBboxPatch(
        (5.45, 5.25), 2.6, 0.75, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#dddddd", edgecolor="#222222", linewidth=1.5,
    ))
    ax.text(6.75, 5.63, "ordinary context\n+ weak CC\n$C_{i,t}$", ha="center", va="center", fontsize=13)

    def arrow(x1, y1, x2, y2, text=None, text_offset=(0, 0.16)):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.7, color="#222222"))
        if text:
            ax.text((x1 + x2) / 2 + text_offset[0], (y1 + y2) / 2 + text_offset[1],
                    text, ha="center", va="center", fontsize=11)

    arrow(2.8, 3.63, 3.35, 3.63)
    arrow(5.75, 3.63, 6.3, 3.63)
    arrow(8.7, 3.63, 9.25, 3.63)
    arrow(6.75, 5.25, 7.5, 4.25, "global cue", (0.3, 0.04))
    ax.text(4.55, 2.45, "$M_{i,t}=Q_B(F_{-i,t-1})$", ha="center", fontsize=16)

    ax.annotate("", xy=(1.25, 3.0), xytext=(10.45, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#555555",
                                connectionstyle="arc3,rad=-0.27"))
    ax.text(6.0, 1.25, "future sensing closes the loop", ha="center", fontsize=12, color="#555555")
    ax.text(6.0, 0.58,
            "$F_{-i,t-1}\;\longrightarrow\;M_{i,t}\;\longrightarrow\;U_{i,t}\;\longrightarrow\;Z_{t+1}$",
            ha="center", fontsize=18)
    ax.set_title("Implemented perception--action causal ordering", fontsize=20, weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "paper_system_diagram_bw.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def functional_plot() -> None:
    data = pd.read_csv(ROOT / "results" / "four_architecture_cc_dropout_summary.csv")
    data = data[data["regime"].isin(["own_sensor", "peer_sensor"])].copy()
    scenarios = ["CC dropout only", "L4 loss only", "L4 loss + CC dropout", "L4 slip only", "L4 slip + CC dropout"]
    labels = ["CC dropout", "L4 loss", "L4 loss + CC", "L4 slip", "L4 slip + CC"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=220)
    x = np.arange(len(scenarios))
    width = 0.34
    styles = {"own_sensor": ("#222222", "///"), "peer_sensor": ("#999999", "\\\\")}
    for offset, regime in [(-width / 2, "own_sensor"), (width / 2, "peer_sensor")]:
        subset = data.set_index(["scenario", "regime"]).loc[[(s, regime) for s in scenarios]]
        ax.bar(x + offset, subset["mean"], width=width, yerr=subset["std"], capsize=4,
               color=styles[regime][0], edgecolor="black", hatch=styles[regime][1],
               label=REGIME_LABEL[regime])
    ax.set_title("Functional robustness under disturbance", fontsize=18, weight="bold")
    ax.set_ylabel("Mean late path error", fontsize=14)
    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "paper_functional_robustness_bw.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def action_plot() -> None:
    data = pd.read_csv(ROOT / "results" / "action_directed_information_corrected.csv")
    data = data[data["history"] == 4].copy()
    data["condition_label"] = data["condition"].map({"intact": "Intact", "limb_loss_sensor_loss_cc_dropout": "Compound"})
    grouped = data.groupby(["regime", "condition_label"])["gain_bits_per_action"].agg(["mean", "std"]).reset_index()
    conditions = ["Intact", "Compound"]
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=220)
    x = np.arange(len(conditions))
    width = 0.34
    styles = {"own_sensor": ("#222222", "///"), "peer_sensor": ("#999999", "\\\\")}
    for offset, regime in [(-width / 2, "own_sensor"), (width / 2, "peer_sensor")]:
        sub = grouped[grouped["regime"] == regime].set_index("condition_label").loc[conditions]
        ax.bar(x + offset, sub["mean"], width=width, yerr=sub["std"], capsize=4,
               color=styles[regime][0], edgecolor="black", hatch=styles[regime][1],
               label=REGIME_LABEL[regime])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Corrected action-directed predictive gain", fontsize=18, weight="bold")
    ax.set_ylabel("Gain $G_{\\mathrm{act}}$ (bits/action)", fontsize=14)
    ax.set_xticks(x, conditions, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "paper_action_gain_bw.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quantization_plot() -> None:
    data = pd.read_csv(ROOT / "results" / "predictor_quantization_itpac.csv")
    data = data[(data["history"] == 4) & (data["levels"] != "full_precision")].copy()
    data["rate"] = data["rate_per_receiver_bits_per_step"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1), constrained_layout=True, dpi=220)
    styles = {
        ("own_sensor", "intact"): ("#222222", "-", "o"),
        ("own_sensor", "limb_loss_sensor_loss_cc_dropout"): ("#555555", "--", "s"),
        ("peer_sensor", "intact"): ("#999999", "-", "^"),
        ("peer_sensor", "limb_loss_sensor_loss_cc_dropout"): ("#000000", "--", "D"),
    }
    names = {
        ("own_sensor", "intact"): "Own-Sensor, intact",
        ("own_sensor", "limb_loss_sensor_loss_cc_dropout"): "Own-Sensor, compound",
        ("peer_sensor", "intact"): "Peer-Sensor, intact",
        ("peer_sensor", "limb_loss_sensor_loss_cc_dropout"): "Peer-Sensor, compound",
    }
    for key, (color, linestyle, marker) in styles.items():
        sub = data[(data["regime"] == key[0]) & (data["condition"] == key[1])].sort_values("rate")
        label = names[key]
        for axis, value, err, title, ylabel in [
            (axes[0], "gain_bits_per_action", None, "Action-predictive gain", "bits/action"),
            (axes[1], "shuffle_gap_bits_per_action", None, "True-message advantage", "bits/action"),
            (axes[2], "eta_per_receiver", None, "Descriptive action efficiency", "predictive bits / nominal bit"),
        ]:
            means = sub.groupby("rate")[value].agg(["mean", "std"]).reset_index()
            axis.errorbar(means["rate"], means["mean"], yerr=means["std"], color=color,
                          linestyle=linestyle, marker=marker, capsize=3, label=label)
    for axis, title, ylabel in zip(axes, ["Action-predictive gain", "True-message advantage", "Descriptive action efficiency"],
                                   ["bits/action", "bits/action", "predictive bits / nominal bit"]):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Per-receiver nominal rate (bits/step)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Predictor-only message quantization: action-directed information", fontsize=16)
    fig.savefig(ROOT / "results" / "paper_predictor_quantization_bw.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    system_diagram()
    functional_plot()
    action_plot()
    quantization_plot()
