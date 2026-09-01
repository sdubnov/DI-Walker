#!/usr/bin/env python3
"""Train frozen policies for Capacity, Act-Comm, and Sensor-Comm."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.policies import save_policy_file
from robust_walker.training import cem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="policies/v4_1b_policies.npz")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(9101, 9107)))
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--pop", type=int, default=36)
    parser.add_argument("--elite", type=int, default=7)
    args = parser.parse_args()

    policies = {}
    rows = []
    for regime in ["capacity", "act_comm", "sensor_comm"]:
        for seed in args.seeds:
            params, loss = cem(regime, seed, iters=args.iters, pop=args.pop, elite=args.elite)
            policies[(regime, seed)] = params
            rows.append({"regime": regime, "seed": seed, "train_loss": loss})
            print(f"{regime} seed={seed} train_loss={loss:.6f}")

    save_policy_file(args.out, policies)
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)
    with (result_dir / "v4_1b_train.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "seed", "train_loss"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
