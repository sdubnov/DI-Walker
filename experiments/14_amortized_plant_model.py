#!/usr/bin/env python3
"""Pilot probabilistic MLP models for closed-loop plant transitions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robust_walker.controllers import CentralControl, Controller
from robust_walker.faults import ForceNoise, LimbFault, SensorFault
from robust_walker.policies import load_policy_file
from robust_walker.simulation import rollout
from robust_walker.tasks import make_task


REGIMES = ["no_sensor", "own_sensor", "peer_sensor", "all_linear"]
TASK_NAMES = ["straight", "s_lr", "s_rl", "sine", "chirp"]
CONDITIONS = {
    "intact": (CentralControl(), LimbFault(), SensorFault()),
    "limb_slip_sensor_retained": (CentralControl(), LimbFault(3, "slip", start_step=65), SensorFault()),
    "limb_loss_sensor_retained": (CentralControl(), LimbFault(3, "loss", strength=0.0, start_step=80), SensorFault()),
    "sensor_failure_only": (CentralControl(), LimbFault(), SensorFault(3, start_step=80)),
    "limb_loss_sensor_loss": (CentralControl(), LimbFault(3, "loss", strength=0.0, start_step=80), SensorFault(3, start_step=80)),
    "limb_loss_sensor_loss_cc_dropout": (
        CentralControl(0.10, "dropout", drop_step=120),
        LimbFault(3, "loss", strength=0.0, start_step=80),
        SensorFault(3, start_step=80),
    ),
}
EVENT_STEPS = {
    "intact": 0,
    "limb_slip_sensor_retained": 65,
    "limb_loss_sensor_retained": 80,
    "sensor_failure_only": 80,
    "limb_loss_sensor_loss": 80,
    "limb_loss_sensor_loss_cc_dropout": 80,
}
MESSAGE_WIDTH = {"no_sensor": 0, "own_sensor": 4, "peer_sensor": 12, "all_linear": 16}
DT = 0.05
FORCE_SCALE = 1.18


def routed_messages(regime: str, sensed: np.ndarray) -> np.ndarray:
    if regime == "no_sensor":
        return np.empty((len(sensed), 0))
    if regime == "own_sensor":
        return sensed.copy()
    if regime == "peer_sensor":
        return np.concatenate([np.delete(sensed, i, axis=1) for i in range(4)], axis=1)
    if regime == "all_linear":
        return np.tile(sensed, (1, 4))
    raise ValueError(regime)


def routed_availability(regime: str, available: np.ndarray) -> np.ndarray:
    return routed_messages(regime, available)


def local_message(regime: str, sensed: np.ndarray, limb: int) -> np.ndarray:
    """Return the message available to one receiver limb."""

    if regime == "no_sensor":
        return np.empty((len(sensed), 0))
    if regime == "own_sensor":
        return sensed[:, limb : limb + 1]
    if regime == "peer_sensor":
        return np.delete(sensed, limb, axis=1)
    if regime == "all_linear":
        return sensed.copy()
    raise ValueError(regime)


def collect_rollout(controller, task, cc, fault, sensor_fault, observation_mode="full", include_availability_mask=True, noise_sigma=0.0, noise_correlation=0.0, noise_seed=0):
    result = rollout(
        controller,
        task,
        central_control=cc,
        limb_fault=fault,
        sensor_fault=sensor_fault,
        force_noise=ForceNoise(
            sigma=noise_sigma,
            correlation=noise_correlation,
            seed=noise_seed,
        ),
    )
    state = result["state_before"]
    activation = result["act_before"]
    error = result["error"]
    sensed = result["message"] / FORCE_SCALE
    available = result["message_available"]
    if observation_mode == "full":
        context = np.column_stack(
            (state, activation, task.vd, task.wd, error, result["cc"])
        )
        message_values = routed_messages(controller.regime.value, sensed)
        message = np.column_stack(
            (message_values, routed_availability(controller.regime.value, available))
        ) if include_availability_mask else message_values
        inputs = np.column_stack((context, message))
        targets = np.column_stack((result["state"], result["act"]))
        return inputs, targets
    if observation_mode == "embodied":
        # One receiver-centric transition per limb. Absolute x/y and the
        # other limbs' activations are intentionally unavailable here.
        rows = []
        targets = []
        for limb in range(4):
            local_context = np.column_stack(
                (
                    state[:, 2:6],
                    activation[:, limb],
                    task.vd,
                    task.wd,
                    error,
                    result["cc"],
                )
            )
            message_values = local_message(controller.regime.value, sensed, limb)
            message = np.column_stack(
                (message_values, local_message(controller.regime.value, available, limb))
            ) if include_availability_mask else message_values
            rows.append(np.column_stack((local_context, message)))
            targets.append(
                np.column_stack((result["act"][:, limb], result["force"][:, limb] / FORCE_SCALE))
            )
        return np.vstack(rows), np.vstack(targets)
    raise ValueError(f"unknown observation mode: {observation_mode}")


def message_width_for(regime: str, observation_mode: str, include_availability_mask: bool) -> int:
    multiplier = 2 if include_availability_mask else 1
    if observation_mode == "full":
        return multiplier * MESSAGE_WIDTH[regime]
    return multiplier * {"no_sensor": 0, "own_sensor": 1, "peer_sensor": 3, "all_linear": 4}[regime]


class GaussianMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64, learn_variance: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.output_dim = output_dim
        self.learn_variance = learn_variance
        self.head = nn.Linear(hidden, (2 if learn_variance else 1) * output_dim)

    def forward(self, x):
        output = self.head(self.body(x))
        if self.learn_variance:
            mean, logvar = output.chunk(2, dim=-1)
            return mean, torch.clamp(logvar, -5.0, 3.0)
        return output, torch.zeros_like(output)


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale


def train_model(train_x, train_y, seed, epochs, hidden, learn_variance):
    torch.manual_seed(seed)
    train_x_mean = train_x.mean(axis=0)
    train_x_scale = train_x.std(axis=0)
    train_x_scale[train_x_scale < 1e-8] = 1.0
    train_y_mean = train_y.mean(axis=0)
    train_y_scale = train_y.std(axis=0)
    train_y_scale[train_y_scale < 1e-8] = 1.0
    train_x = (train_x - train_x_mean) / train_x_scale
    train_y = (train_y - train_y_mean) / train_y_scale
    model = GaussianMLP(train_x.shape[1], train_y.shape[1], hidden, learn_variance)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(train_y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        mean, logvar = model(x)
        loss = 0.5 * (logvar + (y - mean).square() * torch.exp(-logvar)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, (train_x_mean, train_x_scale, train_y_mean, train_y_scale)


def score_model(model, normalization, test_x, test_y):
    train_x_mean, train_x_scale, train_y_mean, train_y_scale = normalization
    test_x = (test_x - train_x_mean) / train_x_scale
    test_y = (test_y - train_y_mean) / train_y_scale
    xt = torch.as_tensor(test_x, dtype=torch.float32)
    yt = torch.as_tensor(test_y, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        mean, logvar = model(xt)
        nll = 0.5 * (logvar + (yt - mean).square() * torch.exp(-logvar)).mean().item()
        rmse = torch.sqrt((yt - mean).square().mean()).item()
    return nll, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", default="policies/four_architecture_policies.npz")
    parser.add_argument("--out", default="results/amortized_plant_model_pilot.csv")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-seeds", type=int, default=6)
    parser.add_argument("--seed-split", type=int, default=4)
    parser.add_argument("--single-seed", type=int, default=None, help="use one policy and split route tasks instead of policy seeds")
    parser.add_argument("--route-split", action="store_true", help="train on the first three route tasks and test on the last two")
    parser.add_argument("--force-noise-sigma", type=float, default=0.0)
    parser.add_argument("--force-noise-correlation", type=float, default=0.0)
    parser.add_argument("--fixed-variance", action="store_true", help="use unit Gaussian variance instead of learning log variance")
    parser.add_argument("--regimes", nargs="*", default=REGIMES)
    parser.add_argument("--observation-mode", choices=["full", "embodied"], default="full")
    parser.add_argument("--include-availability-mask", action="store_true", help="give the estimator explicit sensor-channel availability flags")
    parser.add_argument("--post-event", action="store_true", help="score only transitions at or after each condition's event step")
    args = parser.parse_args()

    policies = load_policy_file(args.policies)
    all_rows = []
    for regime in args.regimes:
        selected = sorted(
            [(seed, params) for (name, seed), params in policies.items() if name == regime]
        )[: args.max_seeds]
        if args.single_seed is not None:
            selected = [(seed, params) for seed, params in selected if seed == args.single_seed]
        if not selected:
            raise ValueError(f"no policy found for {regime}")
        if args.single_seed is None and len(selected) <= args.seed_split:
            raise ValueError("need more policy seeds than --seed-split")
        train_seeds = {seed for seed, _ in selected[: args.seed_split]}
        train_data = {condition: [] for condition in CONDITIONS}
        test_data = {condition: [] for condition in CONDITIONS}
        for seed, params in selected:
            controller = Controller.from_vector(regime, params)
            for task_index, task_name in enumerate(TASK_NAMES):
                if args.single_seed is not None or args.route_split:
                    destination = train_data if task_index < 3 else test_data
                    if args.route_split and seed not in train_seeds:
                        destination = test_data if task_index >= 3 else None
                    elif args.route_split and seed in train_seeds:
                        destination = train_data if task_index < 3 else None
                    if destination is None:
                        continue
                else:
                    destination = train_data if seed in train_seeds else test_data
                task = make_task(task_name)
                for condition_index, (condition, (cc, fault, sensor_fault)) in enumerate(CONDITIONS.items()):
                    x, y = collect_rollout(
                        controller,
                        task,
                        cc,
                        fault,
                        sensor_fault,
                        observation_mode=args.observation_mode,
                        include_availability_mask=args.include_availability_mask,
                        noise_sigma=args.force_noise_sigma,
                        noise_correlation=args.force_noise_correlation,
                        noise_seed=seed + 100 * task_index + condition_index,
                    )
                    destination[condition].append((x, y))

        train_x_all = np.vstack([x for condition in CONDITIONS for x, _ in train_data[condition]])
        train_y_all = np.vstack([y for condition in CONDITIONS for _, y in train_data[condition]])
        message_width = message_width_for(regime, args.observation_mode, args.include_availability_mask)
        context_width = train_x_all.shape[1] - message_width
        context_model, context_norm = train_model(
            train_x_all[:, :context_width], train_y_all,
            seed=42, epochs=args.epochs, hidden=args.hidden,
            learn_variance=not args.fixed_variance,
        )
        message_model, message_norm = train_model(
            train_x_all, train_y_all,
            seed=43, epochs=args.epochs, hidden=args.hidden,
            learn_variance=not args.fixed_variance,
        )
        if message_width == 0:
            # The null architecture must have exactly zero incremental message
            # contribution, not a difference caused by independent training.
            message_model, message_norm = context_model, context_norm
        for condition in CONDITIONS:
            test = test_data[condition]
            test_parts = []
            target_parts = []
            for x, y in test:
                if args.post_event:
                    n_steps = len(y) // (4 if args.observation_mode == "embodied" else 1)
                    time_mask = np.arange(n_steps) >= EVENT_STEPS[condition]
                    if args.observation_mode == "embodied":
                        time_mask = np.tile(time_mask, 4)
                    x, y = x[time_mask], y[time_mask]
                test_parts.append(x)
                target_parts.append(y)
            test_x = np.vstack(test_parts)
            test_y = np.vstack(target_parts)
            context_scores = score_model(
                context_model, context_norm, test_x[:, :context_width], test_y
            )
            message_scores = score_model(message_model, message_norm, test_x, test_y)
            shuffled_x = test_x.copy()
            if message_width:
                permutation = np.random.default_rng(1000 + len(all_rows)).permutation(len(shuffled_x))
                shuffled_x[:, context_width:] = test_x[permutation, context_width:]
            shuffled_scores = score_model(message_model, message_norm, shuffled_x, test_y)
            gain = (context_scores[0] - message_scores[0]) / np.log(2.0)
            for model_name, (nll, rmse) in (
                ("context_only", context_scores),
                ("with_message", message_scores),
                ("with_message_shuffled", shuffled_scores),
            ):
                all_rows.append({
                    "regime": regime,
                    "condition": condition,
                    "model": model_name,
                    "test_nll_nats": nll,
                    "test_rmse_standardized": rmse,
                    "message_gain_bits_per_transition": gain if model_name == "with_message" else "",
                    "variance_model": "fixed" if args.fixed_variance else "learned",
                    "observation_mode": args.observation_mode,
                    "evaluation_window": "post_event" if args.post_event else "full_rollout",
                    "train_seeds": ",".join(map(str, sorted(train_seeds))),
                    "test_seeds": ",".join(map(str, sorted(set(seed for seed, _ in selected) - train_seeds))),
                })
            print(f"{regime:12s} {condition:24s} gain={gain:.5f} bits/transition")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
