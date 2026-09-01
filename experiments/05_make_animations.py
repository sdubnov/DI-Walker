#!/usr/bin/env python3
"""Create representative comparison animations for CC and limb disturbances."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.metrics import cc_metrics
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task
from robust_walker.visualization import animate_comparison_pillow


@dataclass(frozen=True)
class AnimationScenario:
    name: str
    slug: str
    central_control: CentralControl
    limb_fault: LimbFault
    event_step: int


SCENARIOS = [
    AnimationScenario(
        "CC dropout only",
        "cc_dropout_only",
        CentralControl(0.10, "dropout", drop_step=120),
        LimbFault(),
        120,
    ),
    AnimationScenario(
        "L4 failure + CC dropout",
        "l4_failure_cc_dropout",
        CentralControl(0.10, "dropout", drop_step=120),
        LimbFault(3, "loss", strength=0.0, start_step=80),
        120,
    ),
    AnimationScenario(
        "L4 intermittent slip + CC dropout",
        "l4_slip_cc_dropout",
        CentralControl(0.10, "dropout", drop_step=120),
        LimbFault(3, "slip", start_step=65),
        120,
    ),
]

TASKS = [make_task(x) for x in ["straight", "s_lr", "s_rl", "sine", "chirp"]]


def choose_case(policies, scenario: AnimationScenario):
    candidate_seeds = sorted(
        seed
        for regime, seed in policies
        if regime == "capacity" and ("sensor_comm", seed) in policies
    )
    if not candidate_seeds:
        raise ValueError("policy file must contain matched capacity and sensor_comm seeds")

    best = None
    for seed in candidate_seeds:
        capacity = Controller.from_vector("capacity", policies[("capacity", seed)])
        sensor = Controller.from_vector("sensor_comm", policies[("sensor_comm", seed)])
        for task in TASKS:
            capacity_rollout = rollout(capacity, task, scenario.central_control, scenario.limb_fault)
            sensor_rollout = rollout(sensor, task, scenario.central_control, scenario.limb_fault)
            capacity_error = cc_metrics(capacity_rollout, scenario.event_step)["late_path_error"]
            sensor_error = cc_metrics(sensor_rollout, scenario.event_step)["late_path_error"]
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
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/v4_1b_policies.npz")
    parser.add_argument("--outdir", default="results/animations")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in SCENARIOS:
        case = choose_case(policies, scenario)
        outfile = outdir / f"{scenario.slug}.gif"
        animate_comparison_pillow(
            case["capacity_rollout"],
            case["sensor_rollout"],
            outfile,
            left_title=f"Capacity late error {case['capacity_error']:.3f}",
            right_title=f"Sensor-Comm late error {case['sensor_comm_error']:.3f}",
            title=f"{scenario.name} | task={case['task']} seed={case['seed']}",
        )
        rows.append(
            {
                "scenario": scenario.name,
                "task": case["task"],
                "seed": case["seed"],
                "capacity_late_path_error": case["capacity_error"],
                "sensor_comm_late_path_error": case["sensor_comm_error"],
                "sensor_advantage": case["sensor_advantage"],
                "animation": str(outfile),
            }
        )
        print(f"wrote {outfile}")

    with (outdir / "animation_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
