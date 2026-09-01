#!/usr/bin/env python3
"""Create the large V4.3d-style intermittent CC + L4 slip summary GIF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.metrics import cc_metrics
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task
from robust_walker.visualization import animate_spacious_summary_pillow


TASKS = [make_task(x) for x in ["straight", "s_lr", "s_rl", "sine", "chirp"]]


def choose_case(policies):
    scenario_cc = CentralControl(0.10, "intermittent", period=30, on_steps=10)
    scenario_fault = LimbFault(3, "slip", start_step=65)
    candidate_seeds = sorted(
        seed
        for regime, seed in policies
        if regime == "capacity" and ("sensor_comm", seed) in policies
    )
    best = None
    for seed in candidate_seeds:
        capacity = Controller.from_vector("capacity", policies[("capacity", seed)])
        sensor = Controller.from_vector("sensor_comm", policies[("sensor_comm", seed)])
        for task in TASKS:
            capacity_rollout = rollout(capacity, task, central_control=scenario_cc, limb_fault=scenario_fault)
            sensor_rollout = rollout(sensor, task, central_control=scenario_cc, limb_fault=scenario_fault)
            capacity_error = cc_metrics(capacity_rollout, event_step=65)["late_path_error"]
            sensor_error = cc_metrics(sensor_rollout, event_step=65)["late_path_error"]
            advantage = capacity_error - sensor_error
            candidate = {
                "seed": seed,
                "task": task.name,
                "capacity_error": capacity_error,
                "sensor_comm_error": sensor_error,
                "sensor_advantage": advantage,
                "capacity_rollout": capacity_rollout,
                "sensor_rollout": sensor_rollout,
            }
            if best is None or advantage > best["sensor_advantage"]:
                best = candidate
    if best is None:
        raise ValueError("policy file must contain matched capacity and sensor_comm seeds")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/v4_1b_policies.npz")
    parser.add_argument("--out", default="results/animations/v4_3d_large_slow_intermittent_cc_l4_slip_spacious.gif")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    case = choose_case(policies)
    animate_spacious_summary_pillow(
        case["capacity_rollout"],
        case["sensor_rollout"],
        args.out,
        title=f"V4.3d Robust Walker: task={case['task']} seed={case['seed']}",
        left_title="Capacity / No Peer Sensing",
        right_title="Sensor-Comm",
        left_error=case["capacity_error"],
        right_error=case["sensor_comm_error"],
    )

    out = Path(args.out)
    rows = [
        {
            "scenario": "intermittent CC + L4 intermittent slip",
            "task": case["task"],
            "seed": case["seed"],
            "capacity_late_path_error": case["capacity_error"],
            "sensor_comm_late_path_error": case["sensor_comm_error"],
            "sensor_advantage": case["sensor_advantage"],
            "animation": str(out),
        }
    ]
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
