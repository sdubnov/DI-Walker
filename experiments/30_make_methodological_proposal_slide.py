#!/usr/bin/env python3
"""Create a large-font methodological-proposal slide."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.05, 0.93, "A methodological proposal for distributed intelligence", fontsize=30, weight="bold", color="#172033")
    fig.text(0.05, 0.875, "Evaluate information where it changes decisions, then connect it to function through intervention.", fontsize=17, color="#536174")

    boxes = [
        (0.04, 0.49, 0.21, 0.22, "1  Action-directed\ninformation", "$G_{\\mathrm{act}}$\n\nDoes $M_t$ improve\nprediction of $U_t$?", "#e7eef8"),
        (0.28, 0.49, 0.21, 0.22, "2  Intervention", "true / zeroed /\nshuffled / delayed\n\nDoes the message\nchange the action?", "#fff0d9"),
        (0.52, 0.49, 0.21, 0.22, "3  Functional utility", "$J_{\\mathrm{late}}$, $J_{\\mathrm{rec}}$\n$T_{\\mathrm{rec}}$, $P_{\\mathrm{success}}$\n\nDoes the intervention\nchange recovery?", "#e9f5e9"),
        (0.76, 0.49, 0.20, 0.22, "4  Robustness", "disturbance\nenvelopes\nworst-case bounds\n\nCan function be\ncertified?", "#f2e7f8"),
    ]
    for x, y, w, h, title, body, color in boxes:
        fig.patches.append(plt.Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=color, edgecolor="#536174", linewidth=1.5))
        fig.text(x + 0.012, y + h - 0.035, title, fontsize=17, weight="bold", color="#172033", va="top")
        fig.text(x + 0.012, y + h - 0.085, body, fontsize=12.5, color="#172033", va="top", linespacing=1.25)
    for x0, x1 in [(0.25, 0.28), (0.49, 0.52), (0.73, 0.76)]:
        fig.add_artist(plt.Line2D([x0, x1], [0.60, 0.60], transform=fig.transFigure, color="#536174", linewidth=2))
        fig.text((x0 + x1) / 2, 0.60, ">", fontsize=24, color="#536174", ha="center", va="center")

    fig.text(0.05, 0.34, "Modeling gap", fontsize=22, weight="bold", color="#172033")
    fig.text(0.05, 0.285, "A message can be informative for an intermediate action without improving prediction of a later scalar global error.", fontsize=17, color="#536174")
    fig.text(0.05, 0.205, "Therefore, future functional Predictive Information is a useful diagnostic, but not a sufficient standalone measure of distributed intelligence or robustness.", fontsize=17, color="#b45f06")
    fig.text(0.05, 0.105, r"DI-Walker evidence:  Peer messages improve action prediction under compound failure; Peer-Sensor also reduces selected functional errors.", fontsize=16, color="#172033")
    fig.text(0.05, 0.055, "The bridge from information to robustness requires inference-time intervention and disturbance evaluation.", fontsize=16, weight="bold", color="#2878b5")

    fig.savefig(ROOT / "reports" / "di-walker-methodological-proposal-slide.pdf", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "reports" / "di-walker-methodological-proposal-slide.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
