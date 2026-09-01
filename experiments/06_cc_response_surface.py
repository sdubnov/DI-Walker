#!/usr/bin/env python3
"""CC gain and disruption response surface for central-control robustness."""

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


GAINS = [0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 1.0]
CC_MODES = ["always", "dropout", "intermittent", "delayed", "noisy", "stuck"]
TASKS = [make_task(x) for x in ["straight", "s_lr", "s_rl", "sine", "chirp"]]
LIMB_CONDITIONS = [
    ("intact", LimbFault(), 120),
    ("L4 failure", LimbFault(3, "loss", strength=0.0, start_step=80), 120),
    ("L4 slip", LimbFault(3, "slip", start_step=65), 120),
]


def summarize(rows, group_keys, value_key):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(float(row[value_key]))
    out = []
    for key, values in sorted(grouped.items()):
        item = {k: v for k, v in zip(group_keys, key)}
        item["mean"] = mean(values)
        item["std"] = stdev(values) if len(values) > 1 else 0.0
        item["n"] = len(values)
        out.append(item)
    return out


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/v4_1b_policies.npz")
    parser.add_argument("--out", default="results/cc_response_surface.csv")
    parser.add_argument("--acceptable-error", type=float, default=1.0)
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for (regime, seed), params in policies.items():
        if regime not in {"capacity", "sensor_comm"}:
            continue
        controller = Controller.from_vector(regime, params)
        for task in TASKS:
            for gain in GAINS:
                for cc_mode in CC_MODES:
                    cc = CentralControl(gain=gain, mode=cc_mode, drop_step=120, stuck_step=120)
                    for limb_condition, fault, event_step in LIMB_CONDITIONS:
                        r = rollout(controller, task, central_control=cc, limb_fault=fault)
                        rows.append(
                            {
                                "regime": regime,
                                "seed": seed,
                                "task": task.name,
                                "cc_gain": gain,
                                "cc_mode": cc_mode,
                                "limb_condition": limb_condition,
                                **cc_metrics(r, event_step=event_step),
                            }
                        )

    out = Path(args.out)
    write_csv(out, rows)
    summary = summarize(
        rows,
        ["regime", "cc_gain", "cc_mode", "limb_condition"],
        "late_path_error",
    )
    write_csv(out.with_name("cc_response_surface_summary.csv"), summary)

    cmin_rows = []
    for regime in ["capacity", "sensor_comm"]:
        for cc_mode in CC_MODES:
            for limb_condition, _, _ in LIMB_CONDITIONS:
                eligible = [
                    row
                    for row in summary
                    if row["regime"] == regime
                    and row["cc_mode"] == cc_mode
                    and row["limb_condition"] == limb_condition
                    and row["mean"] <= args.acceptable_error
                ]
                cmin = min((row["cc_gain"] for row in eligible), default=None)
                cmin_rows.append(
                    {
                        "regime": regime,
                        "cc_mode": cc_mode,
                        "limb_condition": limb_condition,
                        "acceptable_error": args.acceptable_error,
                        "c_min": cmin if cmin is not None else "",
                    }
                )
    write_csv(out.with_name("cc_minimum_gain.csv"), cmin_rows)

    retention_rows = []
    baseline = {
        (row["regime"], row["seed"], row["task"], row["cc_gain"], row["limb_condition"]): row["late_path_error"]
        for row in rows
        if row["cc_mode"] == "always"
    }
    for row in rows:
        base_key = (row["regime"], row["seed"], row["task"], row["cc_gain"], row["limb_condition"])
        base = baseline.get(base_key)
        if base is None or row["cc_mode"] == "always":
            continue
        retention_rows.append(
            {
                "regime": row["regime"],
                "seed": row["seed"],
                "task": row["task"],
                "cc_gain": row["cc_gain"],
                "cc_mode": row["cc_mode"],
                "limb_condition": row["limb_condition"],
                "always_late_path_error": base,
                "disrupted_late_path_error": row["late_path_error"],
                "retention_ratio": base / row["late_path_error"] if row["late_path_error"] > 0 else "",
            }
        )
    write_csv(out.with_name("cc_disruption_retention.csv"), retention_rows)
    retention_summary = summarize(
        retention_rows,
        ["regime", "cc_gain", "cc_mode", "limb_condition"],
        "retention_ratio",
    )
    write_csv(out.with_name("cc_disruption_retention_summary.csv"), retention_summary)

    for row in summary[:18]:
        print(
            f"{row['regime']:12s} k={row['cc_gain']:<5} {row['cc_mode']:12s} "
            f"{row['limb_condition']:10s} mean={row['mean']:.5f}"
        )


if __name__ == "__main__":
    main()
