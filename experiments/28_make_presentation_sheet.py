#!/usr/bin/env python3
"""Create presentation-ready panels and a 16:9 PDF figure sheet."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def save_panel(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"presentation_{name}.png", dpi=240, bbox_inches="tight", facecolor="white")


def architecture_panel() -> plt.Figure:
    fig = plt.figure(figsize=(11, 5.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.90, "Distributed action pathway", fontsize=26, weight="bold", color="#172033")
    ax.text(0.04, 0.82, "Realized force is available to the next controller update", fontsize=16, color="#536174")

    boxes = [(0.06, 0.54, 0.18, 0.13, "limb j\n$F_{j,t-1}$", "#e7eef8"),
             (0.31, 0.54, 0.18, 0.13, "message\n$M_{j,t}$", "#fff0d9"),
             (0.56, 0.54, 0.18, 0.13, "receiver i\n$U_{i,t}$", "#e9f5e9"),
             (0.81, 0.54, 0.13, 0.13, "plant\n$Z_{t+1}$", "#f2e7f8")]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#536174", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=18, color="#172033")
    for x0, x1 in [(0.24, 0.31), (0.49, 0.56), (0.74, 0.81)]:
        ax.annotate("", xy=(x1, 0.605), xytext=(x0, 0.605), arrowprops=dict(arrowstyle="->", lw=2, color="#536174"))

    ax.text(0.08, 0.33, r"Own-Sensor:  $q_{i,t}=q^{local}_{i,t}+w_i s_{i,t}$", fontsize=19, color="#245d9c")
    ax.text(0.08, 0.23, r"Peer-Sensor:  $q_{i,t}=q^{local}_{i,t}+\sum_{j\ne i}W_{ij}s_{j,t}$", fontsize=19, color="#b45f06")
    ax.text(0.08, 0.10, r"$G_{act}=\mathbb{E}[\log p(U_t|C_t,M_t)-\log p(U_t|C_t)]/\ln 2$", fontsize=17, color="#172033")
    return fig


def robustness_panel() -> plt.Figure:
    rows = list(csv.DictReader((OUT / "adversarial_push_pull_summary.csv").open()))
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    colors = {"own_sensor": "#2878b5", "peer_sensor": "#d97706"}
    labels = {"own_sensor": "Own-Sensor", "peer_sensor": "Peer-Sensor"}
    for regime, color in colors.items():
        selected = sorted(
            [r for r in rows if r["regime"] == regime and abs(float(r["cc_gain"]) - 0.10) < 1e-12],
            key=lambda r: float(r["amplitude"]),
        )
        x = [float(r["amplitude"]) for r in selected]
        y = [float(r["late_path_error_mean"]) for r in selected]
        sd = [float(r["late_path_error_sd"]) for r in selected]
        ax.errorbar(x, y, yerr=sd, marker="o", linewidth=2.5, capsize=4, color=color, label=labels[regime])
    ax.set_title("Adversarial push-pull: deliberate recovery challenge\nL4 + sensor loss", fontsize=17, weight="bold")
    ax.set_xlabel("Push-pull torque amplitude $D$", fontsize=14)
    ax.set_ylabel("Late path error (path units)", fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=13)
    ax.text(0.03, 0.03, "Push-pull = a lateral push followed by an opposite pull.\nIt tests recovery from a transient displacement; Peer-Sensor helps at moderate severity.", transform=ax.transAxes, fontsize=11, color="#536174")
    return fig


def action_gain_panel() -> plt.Figure:
    rows = list(csv.DictReader((OUT / "action_directed_information_corrected.csv").open()))
    selected = [r for r in rows if int(r["history"]) == 4 and r["model"] == "with_message"]
    labels = [("Own-Sensor", "intact"), ("Own-Sensor", "limb_loss_sensor_loss_cc_dropout"),
              ("Peer-Sensor", "intact"), ("Peer-Sensor", "limb_loss_sensor_loss_cc_dropout")]
    means, sds, names, colors = [], [], [], []
    color_map = {"Own-Sensor": "#2878b5", "Peer-Sensor": "#d97706"}
    condition_names = {"intact": "intact", "limb_loss_sensor_loss_cc_dropout": "compound\nfailure"}
    for regime, condition in labels:
        values = [float(r["gain_bits_per_action"]) for r in selected if r["regime"] == regime.lower().replace("-", "_") and r["condition"] == condition]
        means.append(float(np.mean(values)))
        sds.append(float(np.std(values)))
        names.append(f"{regime}\n{condition_names[condition]}")
        colors.append(color_map[regime])
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, means, yerr=sds, capsize=5, color=colors, alpha=0.9)
    ax.axhline(0, color="#172033", linewidth=0.8)
    ax.set_xticks(x, names, fontsize=12)
    ax.set_ylabel("Action-predictive gain (bits/action)", fontsize=14)
    ax.set_title("Action-directed information: $G_{\\mathrm{act}}$", fontsize=18, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.03, 0.94, "Corrected post-event analysis; intact and compound failure (not push-pull).\nIT-PAC-style finite-history estimate of $I(M\\to U)$; error bars show predictor-seed SD.", transform=ax.transAxes, fontsize=9.5, color="#536174", va="top")
    return fig


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panels = [architecture_panel(), robustness_panel(), action_gain_panel()]
    names = ["architecture", "robustness", "action_gain"]
    for fig, name in zip(panels, names):
        save_panel(fig, name)

    sheet = plt.figure(figsize=(16, 9), facecolor="white")
    sheet.text(0.04, 0.955, "DI-Walker: information-guided distributed recovery", fontsize=27, weight="bold", color="#172033")
    sheet.text(0.04, 0.925, "Peer realized-effect communication, robust function, and action-directed predictive information", fontsize=14, color="#536174")
    positions = [(0.04, 0.50, 0.92, 0.38), (0.04, 0.07, 0.44, 0.36), (0.52, 0.07, 0.44, 0.36)]
    for fig, pos in zip(panels, positions):
        canvas = fig.canvas
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())
        ax = sheet.add_axes(pos)
        ax.imshow(image)
        ax.axis("off")
    sheet_path = ROOT / "reports" / "di-walker-presentation-sheet.pdf"
    with PdfPages(sheet_path) as pdf:
        pdf.savefig(sheet, bbox_inches="tight", facecolor="white")
    sheet.savefig(ROOT / "reports" / "di-walker-presentation-sheet.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(sheet)
    print(sheet_path)


if __name__ == "__main__":
    main()
