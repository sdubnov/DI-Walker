#!/usr/bin/env python3
"""Weak central-control dropout and compound disturbance evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.metrics import cc_metrics
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


TASKS = [make_task(x) for x in ["straight", "s_lr", "s_rl", "sine", "chirp"]]
SCENARIOS = [
    ("CC dropout only", CentralControl(0.10, "dropout", drop_step=120), LimbFault(), 120),
    ("CC intermittent only", CentralControl(0.10, "intermittent", drop_step=120), LimbFault(), 120),
    ("L4 loss only", CentralControl(0.10, "always"), LimbFault(3, "loss", strength=0.0, start_step=80), 80),
    ("L4 loss + CC dropout", CentralControl(0.10, "dropout", drop_step=120), LimbFault(3, "loss", strength=0.0, start_step=80), 120),
    ("L4 slip only", CentralControl(0.10, "always"), LimbFault(3, "slip", start_step=65), 65),
    ("L4 slip + CC dropout", CentralControl(0.10, "dropout", drop_step=120), LimbFault(3, "slip", start_step=65), 120),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/v4_1b_policies.npz")
    parser.add_argument("--out", default="results/cc_dropout_results.csv")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for (regime, seed), params in policies.items():
        if regime not in {"capacity", "sensor_comm"}:
            continue
        controller = Controller.from_vector(regime, params)
        for task in TASKS:
            for scenario, cc, fault, event_step in SCENARIOS:
                r = rollout(controller, task, central_control=cc, limb_fault=fault)
                rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "task": task.name,
                        "scenario": scenario,
                        **cc_metrics(r, event_step=event_step),
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
        key = (row["scenario"], row["regime"])
        grouped.setdefault(key, []).append(row["late_path_error"])

    summary_rows = []
    for (scenario, regime), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "scenario": scenario,
                "regime": regime,
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
        )

    with out.with_name("cc_dropout_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "regime", "mean", "std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(f"{row['scenario']:24s} {row['regime']:12s} mean={row['mean']:.5f} std={row['std']:.5f}")


if __name__ == "__main__":
    main()
