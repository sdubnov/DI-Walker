#!/usr/bin/env python3
"""Plot a clear target-versus-trajectory example for the lecture report."""

from __future__ import annotations

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


def main() -> None:
    output = Path("results/didactic_compound_trajectory.png")
    policies = load_policy_file("policies/four_architecture_policies.npz")
    seed = 9104
    task = make_task("s_lr")
    cc = CentralControl(gain=0.10, mode="dropout", drop_step=120)
    fault = LimbFault(limb=3, mode="loss", strength=0.0, start_step=80)
    sensor_fault = SensorFault(limb=3, start_step=80)

    trajectories = {}
    for label, regime in [("Own-Sensor", "own_sensor"), ("Peer-Sensor", "peer_sensor")]:
        controller = Controller.from_vector(regime, policies[(regime, seed)])
        result = rollout(controller, task, central_control=cc, limb_fault=fault, sensor_fault=sensor_fault)
        trajectories[label] = result

    target = task.target_path()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    colors = {"Own-Sensor": "#c44e52", "Peer-Sensor": "#4c72b0"}
    for ax, (label, result) in zip(axes, trajectories.items()):
        ax.plot(target[:, 0], target[:, 1], "k--", linewidth=2.4, label="target route")
        ax.plot(result["state"][:, 0], result["state"][:, 1], color=colors[label], linewidth=2.6, label=label)
        ax.scatter(*target[80, :2], color="#e69f00", s=55, zorder=4, label="L4 loss")
        ax.scatter(*target[120, :2], color="#009e73", s=55, zorder=4, label="CC dropout")
        ax.set_title(label, fontsize=16, weight="bold")
        ax.set_xlabel("body x", fontsize=13)
        ax.set_ylabel("body y", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=10, loc="best")

    fig.suptitle("Compound failure: L4 loss followed by weak-CC dropout", fontsize=18, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
