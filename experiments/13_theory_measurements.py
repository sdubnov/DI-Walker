#!/usr/bin/env python3
"""Estimate finite-time unstable expansion and communication rates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.config import BODY, DT, FORCE_SCALE, TAU
from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.policies import load_policy_file
from robust_walker.tasks import make_task


REGIMES = ["no_sensor", "own_sensor", "peer_sensor", "all_linear"]
TASK_NAMES = ["straight", "s_lr", "s_rl", "sine", "chirp"]
CONDITIONS = {
    "intact": LimbFault(),
    "limb_slip": LimbFault(3, "slip", start_step=65),
    "limb_loss": LimbFault(3, "loss", strength=0.0, start_step=80),
}
CHANNELS = {"no_sensor": 0, "own_sensor": 4, "peer_sensor": 12, "all_linear": 16}


def state_vector(state, activation, sensed_force):
    return np.concatenate((state, activation, sensed_force))


def split_state(z):
    return z[:6], z[6:10], z[10:14]


def step_map(z, controller, task, target_path, t, fault, central_control):
    """One deterministic simulator step, exposed for finite differences."""

    state, activation, sensed_force = split_state(z)
    ph, vx, vy, omega = state[2], state[3], state[4], state[5]
    c, sn = np.cos(ph), np.sin(ph)
    q = controller.logits(task.vd[t], task.wd[t], state, activation, sensed_force)
    target = target_path[t]
    cc_delta, _, _ = central_control.correction(t, state, target, [], None)
    activation = np.clip(activation + DT * (np.tanh(q + cc_delta) - activation) / TAU, -1.0, 1.0)
    force = FORCE_SCALE * activation * fault.multipliers(t)
    total_force = force.sum()
    torque = np.sum(-BODY[:, 1] * force)
    next_state = state.copy()
    next_state[3] += DT * (c * total_force - 0.85 * state[3])
    next_state[4] += DT * (sn * total_force - 0.85 * state[4])
    next_state[5] += DT * (torque / 0.18 - 0.60 * state[5])
    next_state[0] += DT * next_state[3]
    next_state[1] += DT * next_state[4]
    next_state[2] += DT * next_state[5]
    return state_vector(next_state, activation, force)


def jacobian(z, controller, task, target_path, t, fault, central_control, epsilon=1e-5):
    base = step_map(z, controller, task, target_path, t, fault, central_control)
    result = np.empty((len(z), len(z)))
    for k in range(len(z)):
        perturbation = np.zeros_like(z)
        perturbation[k] = epsilon
        result[:, k] = (step_map(z + perturbation, controller, task, target_path, t, fault, central_control) - base) / epsilon
    return result


def expansion_rate(controller, task, fault, horizon):
    """Average positive finite-time singular-value expansion in bits/step."""

    central_control = CentralControl()
    target_path = task.target_path()
    z = np.zeros(14, dtype=float)
    products = []
    jacobians = []
    for t in range(task.n_steps):
        jacobians.append(jacobian(z, controller, task, target_path, t, fault, central_control))
        z = step_map(z, controller, task, target_path, t, fault, central_control)
    for start in range(0, task.n_steps - horizon, horizon):
        product = np.eye(14)
        for t in range(start, start + horizon):
            product = jacobians[t] @ product
        singular_values = np.linalg.svd(product, compute_uv=False)
        products.append(float(np.sum(np.log(np.maximum(singular_values, 1.0))) / (horizon * np.log(2.0))))
    return float(np.mean(products))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/theory_measurements.csv")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--plot", default="results/theory_measurements.png")
    parser.add_argument("--max-seeds", type=int, default=2, help="number of policy seeds used for this numerical diagnostic")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    rows = []
    for regime in REGIMES:
        selected = [(seed, p) for (name, seed), p in policies.items() if name == regime][: args.max_seeds]
        for condition, fault in CONDITIONS.items():
            values = []
            for seed, params in selected:
                controller = Controller.from_vector(regime, params)
                for task_name in TASK_NAMES:
                    values.append(expansion_rate(controller, make_task(task_name), fault, args.horizon))
            if values:
                rows.append({
                    "regime": regime,
                    "condition": condition,
                    "horizon": args.horizon,
                    "expansion_bits_per_step": np.mean(values),
                    "expansion_std": np.std(values),
                    "channel_count": CHANNELS[regime],
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for regime in REGIMES:
        subset = [r for r in rows if r["regime"] == regime]
        x = np.arange(len(subset))
        axes[0].plot(x, [r["expansion_bits_per_step"] for r in subset], "o-", label=regime)
        axes[0].fill_between(x, [r["expansion_bits_per_step"] - r["expansion_std"] for r in subset], [r["expansion_bits_per_step"] + r["expansion_std"] for r in subset], alpha=0.12)
    axes[0].set_xticks(np.arange(len(CONDITIONS)), list(CONDITIONS), rotation=20)
    axes[0].set_ylabel("Finite-time expansion (bits/step)")
    axes[0].set_title("Unstable expansion estimate")
    axes[0].grid(alpha=0.25)

    axes[1].axis("off")
    axes[1].text(0.02, 0.95, "Communication rate per control step", fontsize=13, weight="bold", va="top")
    lines = ["Rate = directed channels x bits/value"]
    for regime in REGIMES:
        lines.append(f"{regime}: {CHANNELS[regime]} channels")
    lines += ["", "The data-rate theorem comparison requires", "an explicit quantized channel and", "a critical expansion estimate."]
    axes[1].text(0.02, 0.82, "\n".join(lines), va="top", family="monospace", fontsize=11)
    fig.suptitle("Theory-facing measurements for DI-Walker", fontsize=15)
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=220)
    print(args.out)
    print(args.plot)
    for row in rows:
        print(f"{row['regime']:12s} {row['condition']:12s} expansion={row['expansion_bits_per_step']:.5f} bits/step")


if __name__ == "__main__":
    main()
