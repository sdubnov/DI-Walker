#!/usr/bin/env python3
"""Create a large-font explanatory push-pull slide and trajectory figure."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault, SensorFault
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def push_pull(n_steps: int, amplitude: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    force = np.zeros(n_steps)
    torque = np.zeros(n_steps)
    force[100:110] = amplitude / 5.0
    force[110:120] = -amplitude / 5.0
    torque[100:110] = amplitude
    torque[110:120] = -amplitude
    return force, torque


def representative_trajectory() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    policies = load_policy_file("policies/four_architecture_policies.npz")
    task = make_task("s_lr")
    force, torque = push_pull(task.n_steps)
    kwargs = dict(
        central_control=CentralControl(0.10, "dropout", drop_step=120),
        limb_fault=LimbFault(3, "loss", strength=0.0, start_step=80),
        sensor_fault=SensorFault(3, start_step=80),
        external_force_y=force,
        external_torque=torque,
    )
    own = rollout(Controller.from_vector("own_sensor", policies[("own_sensor", 9101)]), task, **kwargs)
    peer = rollout(Controller.from_vector("peer_sensor", policies[("peer_sensor", 9101)]), task, **kwargs)
    return own, peer, task.target_path()


def make_slide() -> None:
    own, peer, target = representative_trajectory()
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.04, 0.945, "Why use an adversarial push-pull disturbance?", fontsize=29, weight="bold", color="#172033")
    fig.text(0.04, 0.895, "A static limb failure can create a stable but incorrect trajectory. Push-pull tests recovery from a deliberately escalating disturbance.", fontsize=15, color="#536174")

    ax = fig.add_axes([0.05, 0.18, 0.48, 0.62])
    ax.plot(target[:, 0], target[:, 1], "k--", linewidth=2.5, label="target path")
    ax.plot(own["state"][:, 0], own["state"][:, 1], color="#2878b5", linewidth=2.5, label="Own-Sensor")
    ax.plot(peer["state"][:, 0], peer["state"][:, 1], color="#d97706", linewidth=2.5, label="Peer-Sensor")
    for t, label in [(80, "L4 + sensor loss"), (100, "push"), (120, "opposite pull / CC dropout")]:
        ax.scatter(own["state"][t, 0], own["state"][t, 1], s=55, color="#172033", zorder=3)
        ax.text(own["state"][t, 0], own["state"][t, 1], "  " + label, fontsize=10, color="#172033")
    ax.set_title("Representative trajectory, D = 2", fontsize=19, weight="bold")
    ax.set_xlabel("x position", fontsize=14)
    ax.set_ylabel("y position", fontsize=14)
    ax.legend(frameon=False, fontsize=13)
    ax.grid(alpha=0.25)

    ax2 = fig.add_axes([0.60, 0.51, 0.35, 0.29])
    rows = list(csv.DictReader((OUT / "adversarial_push_pull_summary.csv").open()))
    colors = {"own_sensor": "#2878b5", "peer_sensor": "#d97706"}
    labels = {"own_sensor": "Own-Sensor", "peer_sensor": "Peer-Sensor"}
    for regime, color in colors.items():
        selected = sorted([r for r in rows if r["regime"] == regime and abs(float(r["cc_gain"]) - 0.10) < 1e-12], key=lambda r: float(r["amplitude"]))
        ax2.plot([float(r["amplitude"]) for r in selected], [float(r["late_path_error_mean"]) for r in selected], "o-", linewidth=2.2, color=color, label=labels[regime])
    ax2.set_title("Population summary: late error", fontsize=17, weight="bold")
    ax2.set_xlabel("push-pull severity D", fontsize=12)
    ax2.set_ylabel("path units", fontsize=12)
    ax2.legend(frameon=False, fontsize=11)
    ax2.grid(alpha=0.25)

    fig.text(0.60, 0.43, "What the scalar graph shows", fontsize=18, weight="bold", color="#172033", va="top")
    fig.text(0.60, 0.385, "It averages the final tracking deviation over many routes,\npolicy seeds, and push directions. It is a population-level\nrobustness summary, not a picture of balance recovery.", fontsize=14, color="#536174", va="top")
    fig.text(0.60, 0.225, "What push-pull adds", fontsize=18, weight="bold", color="#172033", va="top")
    fig.text(0.60, 0.18, "The first pulse displaces or rotates the body; the opposite pulse reverses\nthe disturbance. This probes whether the surviving controller can recover\nfrom an adversarial loss-of-balance event.", fontsize=14, color="#536174", va="top")
    fig.text(0.60, 0.055, "This is robustness to an adversarial disturbance, not a proof of arbitrary unstable-system stabilization.", fontsize=13, color="#b45f06", va="top")

    fig.savefig(ROOT / "reports" / "di-walker-push-pull-explanation-slide.pdf", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "reports" / "di-walker-push-pull-explanation-slide.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "presentation_push_pull_trajectory.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_slide()
