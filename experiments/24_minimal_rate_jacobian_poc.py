#!/usr/bin/env python3
"""Minimal frozen-policy rate sweep and finite-time expansion diagnostic.

This is a didactic comparison, not a data-rate-theorem proof.  It holds one
Peer-Sensor policy fixed, sweeps the precision of its realized-force message,
and compares tracking degradation with a local residual expansion estimate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.config import BODY, DT, FORCE_SCALE, TAU
from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


LEVELS = [1, 3, 5, 9, 17, 33, 65]
STATE_SCALES = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])
ERROR_SCALES = np.array([0.5, 0.5, 0.5, 0.25])


def state_vector(state: np.ndarray, activation: np.ndarray) -> np.ndarray:
    return np.concatenate((state, activation))


def split_state(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return z[:6], z[6:10]


def residual_step_map(
    z: np.ndarray,
    t: int,
    task,
    controller: Controller,
    fault: LimbFault,
    nominal_message: np.ndarray,
) -> np.ndarray:
    """Advance the deterministic plant with the peer message held fixed."""

    state, activation = split_state(z)
    q = controller.logits(task.vd[t], task.wd[t], state, activation, nominal_message)
    activation_next = np.clip(
        activation + DT * (np.tanh(q) - activation) / TAU,
        -1.0,
        1.0,
    )
    force = FORCE_SCALE * activation_next * fault.multipliers(t)
    c, sn = np.cos(state[2]), np.sin(state[2])
    total_force = force.sum()
    torque = np.sum(-BODY[:, 1] * force)
    next_state = state.copy()
    next_state[3] += DT * (c * total_force - 0.85 * state[3])
    next_state[4] += DT * (sn * total_force - 0.85 * state[4])
    next_state[5] += DT * (torque / 0.18 - 0.60 * state[5])
    next_state[0] += DT * next_state[3]
    next_state[1] += DT * next_state[4]
    next_state[2] += DT * next_state[5]
    return state_vector(next_state, activation_next)


def finite_difference_jacobian(step_fn, z: np.ndarray, relative_eps: float = 1e-5) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    base = np.asarray(step_fn(z))
    jacobian = np.empty((base.size, z.size), dtype=float)
    for column in range(z.size):
        h = relative_eps * max(1.0, abs(z[column]))
        plus = z.copy()
        minus = z.copy()
        plus[column] += h
        minus[column] -= h
        jacobian[:, column] = (step_fn(plus) - step_fn(minus)) / (2.0 * h)
    return jacobian


def scale_jacobian(jacobian: np.ndarray) -> np.ndarray:
    return (jacobian * STATE_SCALES[None, :]) / STATE_SCALES[:, None]


def jacobian_product(jacobians: list[np.ndarray], start: int, horizon: int) -> np.ndarray:
    product = np.eye(jacobians[0].shape[0])
    for jacobian in jacobians[start : start + horizon]:
        product = jacobian @ product
    return product


def critical_rate(jacobians: list[np.ndarray], start: int, horizon: int) -> float:
    product = jacobian_product(jacobians, start, horizon)
    singular_values = np.linalg.svd(product, compute_uv=False)
    expanding = singular_values[singular_values > 1.0]
    return float(np.log(expanding).sum() / (horizon * np.log(2.0))) if expanding.size else 0.0


def tracking_vector(result: dict[str, np.ndarray]) -> np.ndarray:
    state = result["state"]
    target = result["target"]
    dx = target[:, 0] - state[:, 0]
    dy = target[:, 1] - state[:, 1]
    e_lat = -np.sin(state[:, 2]) * dx + np.cos(state[:, 2]) * dy
    e_heading = np.arctan2(
        np.sin(target[:, 2] - state[:, 2]),
        np.cos(target[:, 2] - state[:, 2]),
    )
    e_velocity = result["error"][:, 0]
    e_yaw_rate = result["error"][:, 2]
    return np.column_stack((e_lat, e_heading, e_velocity, e_yaw_rate))


def normalized_error(result: dict[str, np.ndarray]) -> np.ndarray:
    errors = tracking_vector(result) / ERROR_SCALES[None, :]
    return np.sum(errors * errors, axis=1)


def expansion_summary(
    result: dict[str, np.ndarray],
    task,
    controller: Controller,
    fault: LimbFault,
    horizon: int,
    start_step: int,
    settling_steps: int,
) -> tuple[float, float, float]:
    jacobians = []
    for t, (state, activation, message) in enumerate(
        zip(result["state_before"], result["act_before"], result["message"])
    ):
        z = state_vector(state, activation)
        fixed_message = message.copy()
        step_fn = lambda candidate, t=t, fixed_message=fixed_message: residual_step_map(
            candidate, t, task, controller, fault, fixed_message
        )
        jacobians.append(scale_jacobian(finite_difference_jacobian(step_fn, z)))

    first = start_step + settling_steps
    starts = range(first, len(jacobians) - horizon + 1)
    rates = [critical_rate(jacobians, start, horizon) for start in starts]
    if not rates:
        raise ValueError("trajectory is too short for the requested Jacobian window")
    return float(np.median(rates)), float(np.quantile(rates, 0.10)), float(np.quantile(rates, 0.90))


def summarize_rollout(result: dict[str, np.ndarray], late_start: int) -> tuple[float, float, float]:
    values = normalized_error(result)
    late_values = values[late_start:]
    times = np.arange(late_start, len(values), dtype=float)
    slope = float(np.polyfit(times, np.log1p(late_values), 1)[0])
    return float(np.mean(late_values)), slope, float(np.mean(result["message_clip_fraction"][late_start:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--fault-step", type=int, default=200)
    parser.add_argument("--settling-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--out", default="results/minimal_rate_jacobian_poc.csv")
    parser.add_argument("--plot", default="results/minimal_rate_jacobian_poc.png")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    controller = Controller.from_vector("peer_sensor", policies[("peer_sensor", args.seed)])
    task = make_task("straight", n_steps=args.steps)
    conditions = {
        "intact": LimbFault(),
        "L4_loss": LimbFault(3, "loss", strength=0.0, start_step=args.fault_step),
    }
    central_control = CentralControl()
    rows = []

    for condition, fault in conditions.items():
        full = rollout(controller, task, central_control=central_control, limb_fault=fault)
        rcrit = expansion_summary(
            full, task, controller, fault, args.horizon, args.fault_step if condition != "intact" else 0, args.settling_steps
        )
        late_start = args.fault_step + args.settling_steps
        full_mse, full_slope, full_clip = summarize_rollout(full, late_start)
        for n_levels in LEVELS:
            result = rollout(
                controller,
                task,
                central_control=central_control,
                limb_fault=fault,
                sensor_levels=n_levels,
            )
            mse, slope, clip_fraction = summarize_rollout(result, late_start)
            rows.append(
                {
                    "condition": condition,
                    "n_levels": n_levels,
                    "rate_bits_per_step": 4.0 * np.log2(n_levels),
                    "late_tracking_mse": mse,
                    "late_log_error_slope": slope,
                    "full_precision_late_tracking_mse": full_mse,
                    "full_precision_late_log_error_slope": full_slope,
                    "rcrit_median": rcrit[0],
                    "rcrit_q10": rcrit[1],
                    "rcrit_q90": rcrit[2],
                    "quantizer_clip_fraction": clip_fraction,
                    "late_start": late_start,
                    "horizon": args.horizon,
                    "policy_seed": args.seed,
                }
            )
        rows.append(
            {
                "condition": condition,
                "n_levels": "full_precision",
                "rate_bits_per_step": "",
                "late_tracking_mse": full_mse,
                "late_log_error_slope": full_slope,
                "full_precision_late_tracking_mse": full_mse,
                "full_precision_late_log_error_slope": full_slope,
                "rcrit_median": rcrit[0],
                "rcrit_q10": rcrit[1],
                "rcrit_q90": rcrit[2],
                "quantizer_clip_fraction": full_clip,
                "late_start": late_start,
                "horizon": args.horizon,
                "policy_seed": args.seed,
            }
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for condition, color in [("intact", "tab:blue"), ("L4_loss", "tab:orange")]:
        subset = [row for row in rows if row["condition"] == condition and row["n_levels"] != "full_precision"]
        x = np.array([float(row["rate_bits_per_step"]) for row in subset])
        y = np.array([float(row["late_tracking_mse"]) for row in subset])
        axes[0].plot(x, y, "o-", color=color, label=condition)
        rcrit = float(subset[0]["rcrit_median"])
        q10 = float(subset[0]["rcrit_q10"])
        q90 = float(subset[0]["rcrit_q90"])
        axes[0].axvline(rcrit, color=color, linestyle="--", alpha=0.8)
        axes[0].axvspan(q10, q90, color=color, alpha=0.10)
        full = next(row for row in rows if row["condition"] == condition and row["n_levels"] == "full_precision")
        axes[0].scatter([x.max() * 1.08], [float(full["late_tracking_mse"])], color=color, marker="*", s=100)

        slopes = np.array([float(row["late_log_error_slope"]) for row in subset])
        axes[1].plot(x, slopes, "o-", color=color, label=condition)
    zoom = inset_axes(axes[0], width="38%", height="31%", loc="lower left", borderpad=1.4)
    zoom.set_xlim(0.0, 0.08)
    zoom.set_ylim(0.0, 0.04)
    for condition, color in [("intact", "tab:blue"), ("L4_loss", "tab:orange")]:
        subset = [row for row in rows if row["condition"] == condition and row["n_levels"] != "full_precision"]
        rcrit = float(subset[0]["rcrit_median"])
        q10 = float(subset[0]["rcrit_q10"])
        q90 = float(subset[0]["rcrit_q90"])
        zoom.axvspan(q10, q90, color=color, alpha=0.18)
        zoom.axvline(rcrit, color=color, linestyle="--", linewidth=1.4)
        zoom.text(rcrit + 0.002, 0.033 if condition == "intact" else 0.025, condition, color=color, fontsize=7)
    zoom.set_title("Critical-rate scale", fontsize=8)
    zoom.set_xlabel("bits/step", fontsize=7)
    zoom.set_ylabel("", fontsize=7)
    zoom.tick_params(labelsize=7)
    zoom.grid(alpha=0.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Nominal shared-bus rate (bits/step)")
    axes[0].set_ylabel("Late normalized tracking MSE")
    axes[0].set_title("Quantization and functional error")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Nominal shared-bus rate (bits/step)")
    axes[1].set_ylabel("Slope of log(1 + V)")
    axes[1].set_title("Late error growth")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Minimal rate/Jacobian proof of concept: frozen Peer-Sensor policy")
    figure.savefig(args.plot, dpi=220)
    print(f"CSV: {output}")
    print(f"Plot: {args.plot}")
    for condition in conditions:
        row = next(row for row in rows if row["condition"] == condition and row["n_levels"] == LEVELS[0])
        print(
            f"{condition}: Rcrit={row['rcrit_median']:.5f} "
            f"[{row['rcrit_q10']:.5f}, {row['rcrit_q90']:.5f}] bits/step"
        )


if __name__ == "__main__":
    main()
