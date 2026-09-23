#!/usr/bin/env python3
"""Frozen-policy push-pull disturbance and optional CC-substitution experiment.

The plant receives a pre-registered lateral push followed by an opposite push.
The learned controller, policy archive, route set, and sensor architecture are
unchanged. The optional CC sweep estimates how much weak global supervision is
needed for recovery, rather than treating CC as an information-rate estimate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import LimbFault, SensorFault
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import ID_TASKS, OOS_TASKS


REGIMES = ["own_sensor", "peer_sensor"]
POLICY_SEEDS = [9101, 9102, 9103, 9104, 9105, 9106]
CC_GAINS = [0.0, 0.05, 0.10, 0.20]
AMPLITUDES = [1.0, 2.0, 5.0, 8.0]
EVENT_STEP = 80
PUSH_STEP = 100
PULSE_STEPS = 10
FORCE_SIGNS = [1.0, -1.0]


def push_pull(
    n_steps: int, amplitude: float, sign: float, push_step: int = PUSH_STEP
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired lateral-force and torque +D/-D pulses.

    The torque amplitude is the sweep variable. The lateral-force amplitude is
    one fifth of it, keeping the two perturbations on comparable pilot scales.
    """

    force = np.zeros(n_steps, dtype=float)
    torque = np.zeros(n_steps, dtype=float)
    force[push_step : push_step + PULSE_STEPS] = sign * amplitude / 5.0
    force[push_step + PULSE_STEPS : push_step + 2 * PULSE_STEPS] = -sign * amplitude / 5.0
    torque[push_step : push_step + PULSE_STEPS] = sign * amplitude
    torque[push_step + PULSE_STEPS : push_step + 2 * PULSE_STEPS] = -sign * amplitude
    return force, torque


def metrics(result: dict[str, np.ndarray], event_step: int, threshold: float = 3.5) -> dict[str, float]:
    """Compute recovery, error-growth, and success metrics."""

    error = np.asarray(result["path_error"], dtype=float)
    post = error[event_step:]
    late_start = max(event_step + 40, len(error) - 80)
    late = error[late_start:]
    time = np.arange(event_step, len(error), dtype=float)
    slope = float(np.polyfit(time, np.log1p(post), 1)[0]) if len(post) > 1 else float("nan")
    pulse_end = PUSH_STEP + 2 * PULSE_STEPS
    baseline = float(np.mean(error[event_step:PUSH_STEP]))
    peak_excess = max(float(np.max(error[PUSH_STEP:]) - baseline), 0.0)
    recovery_level = baseline + 0.20 * peak_excess
    recovery_horizon = float(len(error) - pulse_end)
    for t in range(pulse_end, len(error)):
        if np.all(error[t:] <= recovery_level):
            recovery_horizon = float(t - pulse_end)
            break
    recovery_window = error[pulse_end:]
    recovery_time = np.arange(len(recovery_window), dtype=float)
    excess = np.maximum(recovery_window - baseline, 1e-3)
    recovery_rate = float(-np.polyfit(recovery_time, np.log(excess), 1)[0]) if len(excess) > 1 else float("nan")
    recovery_time = float(len(error) - event_step)
    for t in range(event_step, len(error)):
        if np.all(error[t:] <= threshold):
            recovery_time = float(t - event_step)
            break
    return {
        "peak_path_error": float(np.max(post)),
        "late_path_error": float(np.mean(late)),
        "recovery_integral": float(np.sum(post) * 0.05),
        "recovery_time_steps": recovery_time,
        "error_growth_slope": slope,
        "pre_push_baseline_error": baseline,
        "peak_excess_error": peak_excess,
        "recovery_horizon_20pct_steps": recovery_horizon,
        "recovery_rate_per_step": recovery_rate,
        "success": float(np.mean(late) <= threshold),
    }


def counterfactual_metrics(
    disturbed: dict[str, np.ndarray],
    no_push: dict[str, np.ndarray],
    pulse_end: int = PUSH_STEP + 2 * PULSE_STEPS,
) -> dict[str, float]:
    """Measure incremental push response relative to a matched no-push rollout."""

    delta = np.abs(disturbed["path_error"] - no_push["path_error"])
    post = delta[pulse_end:]
    peak = float(np.max(post))
    level = 0.20 * peak
    horizon = float(len(delta) - pulse_end)
    for t in range(pulse_end, len(delta)):
        if np.all(delta[t:] <= level):
            horizon = float(t - pulse_end)
            break
    excess = np.maximum(post, 1e-3)
    time = np.arange(len(post), dtype=float)
    rate = float(-np.polyfit(time, np.log(excess), 1)[0]) if len(excess) > 1 else float("nan")
    return {
        "counterfactual_peak_deviation": peak,
        "counterfactual_integral": float(np.sum(post) * 0.05),
        "counterfactual_recovery_horizon_20pct_steps": horizon,
        "counterfactual_recovery_rate_per_step": rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/adversarial_push_pull.csv")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-cc-sweep", action="store_true")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    regimes = REGIMES
    seeds = POLICY_SEEDS
    tasks = ID_TASKS + OOS_TASKS
    amplitudes = AMPLITUDES
    signs = FORCE_SIGNS
    gains = [0.10] if args.no_cc_sweep else CC_GAINS
    if args.smoke:
        regimes, seeds, tasks, amplitudes, signs, gains = ["own_sensor"], [9101], ID_TASKS[:1], [1.0], [1.0], [0.10]

    rows: list[dict[str, object]] = []
    for regime in regimes:
        for seed in seeds:
            controller = Controller.from_vector(regime, policies[(regime, seed)])
            for task in tasks:
                for amplitude in amplitudes:
                    for sign in signs:
                        disturbance_force, disturbance_torque = push_pull(task.n_steps, amplitude, sign)
                        for cc_gain in gains:
                            cc = CentralControl(gain=cc_gain, mode="dropout", drop_step=120)
                            fault = LimbFault(limb=3, mode="loss", strength=0.0, start_step=EVENT_STEP)
                            sensor_fault = SensorFault(limb=3, start_step=EVENT_STEP)
                            result = rollout(
                                controller,
                                task,
                                central_control=cc,
                                limb_fault=fault,
                                sensor_fault=sensor_fault,
                                external_force_y=disturbance_force,
                                external_torque=disturbance_torque,
                            )
                            no_push = rollout(
                                controller,
                                task,
                                central_control=cc,
                                limb_fault=fault,
                                sensor_fault=sensor_fault,
                            )
                            rows.append(
                                {
                                    "regime": regime,
                                    "seed": seed,
                                    "task": task.name,
                                    "amplitude": amplitude,
                                    "sign": sign,
                                    "cc_gain": cc_gain,
                                    "condition": "L4_loss_sensor_loss_CC_dropout_push_pull",
                                    **metrics(result, EVENT_STEP),
                                    **counterfactual_metrics(result, no_push),
                                }
                            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["regime"], row["amplitude"], row["cc_gain"])
        summary.setdefault(key, []).append(row)
    summary_rows = []
    for (regime, amplitude, cc_gain), values in sorted(summary.items()):
        summary_rows.append(
            {
                "regime": regime,
                "amplitude": amplitude,
                "cc_gain": cc_gain,
                "n": len(values),
                "late_path_error_mean": np.mean([v["late_path_error"] for v in values]),
                "late_path_error_sd": np.std([v["late_path_error"] for v in values]),
                "recovery_integral_mean": np.mean([v["recovery_integral"] for v in values]),
                "recovery_time_mean": np.mean([v["recovery_time_steps"] for v in values]),
                "recovery_horizon_20pct_mean": np.mean([v["recovery_horizon_20pct_steps"] for v in values]),
                "recovery_rate_mean": np.mean([v["recovery_rate_per_step"] for v in values]),
                "counterfactual_peak_deviation_mean": np.mean([v["counterfactual_peak_deviation"] for v in values]),
                "counterfactual_integral_mean": np.mean([v["counterfactual_integral"] for v in values]),
                "counterfactual_horizon_20pct_mean": np.mean([v["counterfactual_recovery_horizon_20pct_steps"] for v in values]),
                "counterfactual_rate_mean": np.mean([v["counterfactual_recovery_rate_per_step"] for v in values]),
                "error_growth_slope_mean": np.mean([v["error_growth_slope"] for v in values]),
                "success_rate": np.mean([v["success"] for v in values]),
            }
        )
    summary_path = out.with_name(out.stem + "_summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_path = out.with_suffix(".png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {"own_sensor": "tab:blue", "peer_sensor": "tab:orange"}
    for regime in regimes:
        for cc_gain in [0.10]:
            selected = [
                row for row in summary_rows
                if row["regime"] == regime and abs(float(row["cc_gain"]) - cc_gain) < 1e-12
            ]
            selected.sort(key=lambda row: float(row["amplitude"]))
            x = [row["amplitude"] for row in selected]
            axes[0].plot(x, [row["late_path_error_mean"] for row in selected], "o-", color=colors[regime], label=regime)
            axes[1].plot(x, [row["recovery_integral_mean"] for row in selected], "o-", color=colors[regime], label=regime)
            axes[2].plot(x, [row["success_rate"] for row in selected], "o-", color=colors[regime], label=regime)
    axes[0].set_ylabel("late path error")
    axes[1].set_ylabel("post-event error integral")
    axes[2].set_ylabel("success probability")
    for axis in axes:
        axis.set_xlabel("push-pull torque amplitude")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[2].set_ylim(-0.02, 1.02)
    fig.suptitle("Adversarial push-pull after L4 and sensor loss; CC gain = 0.10")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    for row in summary_rows:
        print(
            f"{row['regime']:11s} D={row['amplitude']:.2f} CC={row['cc_gain']:.2f} "
            f"late={row['late_path_error_mean']:.4f} recovery={row['recovery_integral_mean']:.4f} "
            f"slope={row['error_growth_slope_mean']:.5f} success={row['success_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
