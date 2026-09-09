#!/usr/bin/env python3
"""Rate-distortion sweep using quantized realized-force messages."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.metrics import cc_metrics, tracking_loss
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


REGIMES = ["no_sensor", "own_sensor", "peer_sensor", "all_linear"]
TASKS = [make_task(name) for name in ["straight", "s_lr", "s_rl", "sine", "chirp"]]
SCENARIOS = {
    "limb_slip": (CentralControl(0.10, "always"), LimbFault(3, "slip", start_step=65)),
    "limb_loss_cc_dropout": (
        CentralControl(0.10, "dropout", drop_step=120),
        LimbFault(3, "loss", strength=0.0, start_step=80),
    ),
}
CHANNELS = {"no_sensor": 0, "own_sensor": 4, "peer_sensor": 12, "all_linear": 16}
BITS = [0, 1, 2, 4, 8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/quantized_message_sweep.csv")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for regime in REGIMES:
        selected = [(seed, params) for (name, seed), params in policies.items() if name == regime]
        if not selected:
            continue
        for scenario, (cc, fault) in SCENARIOS.items():
            for bits in BITS:
                for seed, params in selected:
                    controller = Controller.from_vector(regime, params)
                    for task in TASKS:
                        result = rollout(controller, task, central_control=cc, limb_fault=fault, sensor_bits=bits)
                        rate = CHANNELS[regime] * bits / 0.05
                        rows.append({
                            "regime": regime,
                            "seed": seed,
                            "task": task.name,
                            "scenario": scenario,
                            "bits_per_value": bits,
                            "channel_count": CHANNELS[regime],
                            "rate_bits_per_second": rate,
                            "tracking_loss": tracking_loss(result),
                            **cc_metrics(result, event_step=65 if scenario == "limb_slip" else 120),
                        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for regime in REGIMES:
        for scenario in SCENARIOS:
            for bits in BITS:
                values = [r["late_path_error"] for r in rows if r["regime"] == regime and r["scenario"] == scenario and r["bits_per_value"] == bits]
                if values:
                    rate = CHANNELS[regime] * bits / 0.05
                    print(f"{regime:12s} {scenario:24s} bits={bits} rate={rate:5.0f} late_path_error={np.mean(values):.4f}")


if __name__ == "__main__":
    main()
