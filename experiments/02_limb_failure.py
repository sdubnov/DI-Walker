#!/usr/bin/env python3
"""Evaluate frozen policies under held-out limb weakening and failure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import Controller
from robust_walker.faults import LimbFault
from robust_walker.metrics import tracking_loss
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import ID_TASKS, OOS_TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/v4_1b_policies.npz")
    parser.add_argument("--out", default="results/limb_failure_results.csv")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for (regime, seed), params in policies.items():
        controller = Controller.from_vector(regime, params)
        for split, tasks in [("ID", ID_TASKS), ("OOS", OOS_TASKS)]:
            for task in tasks:
                intact = tracking_loss(rollout(controller, task))
                rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "split": split,
                        "task": task.name,
                        "condition": "intact",
                        "limb": 0,
                        "retained_strength": 1.0,
                        "loss": intact,
                        "excess_loss": 0.0,
                    }
                )
                for limb in range(4):
                    for strength in [0.5, 0.0]:
                        fault = LimbFault(limb=limb, mode="loss", strength=strength, start_step=80)
                        loss = tracking_loss(rollout(controller, task, limb_fault=fault))
                        rows.append(
                            {
                                "regime": regime,
                                "seed": seed,
                                "split": split,
                                "task": task.name,
                                "condition": f"L{limb + 1}",
                                "limb": limb + 1,
                                "retained_strength": strength,
                                "loss": loss,
                                "excess_loss": loss - intact,
                            }
                        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        if row["condition"] == "intact":
            continue
        key = (row["regime"], row["split"], row["retained_strength"])
        grouped.setdefault(key, []).append(row["excess_loss"])

    summary_rows = []
    for (regime, split, retained_strength), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "regime": regime,
                "split": split,
                "retained_strength": retained_strength,
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
        )

    with out.with_name("limb_failure_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "split", "retained_strength", "mean", "std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            f"{row['regime']:12s} {row['split']:3s} strength={row['retained_strength']:.1f} "
            f"mean={row['mean']:.5f} std={row['std']:.5f}"
        )


if __name__ == "__main__":
    main()
