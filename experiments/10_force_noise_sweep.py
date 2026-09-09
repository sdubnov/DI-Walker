#!/usr/bin/env python3
"""Evaluate architecture resilience to independent realized-force noise."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import Controller
from robust_walker.faults import ForceNoise
from robust_walker.metrics import cc_metrics, tracking_loss
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


REGIMES = ["no_sensor", "own_sensor", "peer_sensor", "all_linear"]
TASKS = [make_task(name) for name in ["straight", "s_lr", "s_rl", "sine", "chirp"]]
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/force_noise_sweep.csv")
    parser.add_argument("--correlation", type=float, default=0.0)
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for regime in REGIMES:
        selected = [(seed, params) for (name, seed), params in policies.items() if name == regime]
        if not selected:
            continue
        for sigma in SIGMAS:
            for seed, params in selected:
                controller = Controller.from_vector(regime, params)
                for task in TASKS:
                    result = rollout(
                        controller,
                        task,
                        force_noise=ForceNoise(sigma=sigma, correlation=args.correlation, seed=seed),
                    )
                    rows.append({
                        "regime": regime,
                        "seed": seed,
                        "task": task.name,
                        "sigma": sigma,
                        "correlation": args.correlation,
                        "tracking_loss": tracking_loss(result),
                        **cc_metrics(result, event_step=0),
                    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for regime in REGIMES:
        for sigma in SIGMAS:
            values = [r["late_path_error"] for r in rows if r["regime"] == regime and r["sigma"] == sigma]
            if values:
                print(f"{regime:12s} sigma={sigma:.2f} late_path_error={sum(values) / len(values):.4f}")


if __name__ == "__main__":
    main()
