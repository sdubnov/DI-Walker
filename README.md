# DI-Walker: Distributed Intelligence and Failure Recovery

Working folder for the October 23 ELLIIT symposium talk on robust AI, distributed sensory information, and information-theoretic measures such as predictive information, directed information, transfer entropy, and rate-distortion-style functional information efficiency.

The refactored code in this folder is derived from the local Robust Walker experiments:

`/Users/sdubnov/Documents/Research/Work/DI and ICL/Octopus/Robust Walker`

The theoretical context is drawn from:

`/Users/sdubnov/Documents/Research/Talks/ELLIIT`

## Scientific Separation

The repository separates:

- `robust_walker/config.py`: frozen physical constants and body geometry.
- `robust_walker/tasks.py`: goal-conditioned trajectory distributions.
- `robust_walker/controllers.py`: Capacity, Act-Comm, Sensor-Comm, and weak central control.
- `robust_walker/faults.py`: limb failure and intermittent slip disturbances.
- `robust_walker/simulation.py`: one canonical rollout implementation.
- `robust_walker/training.py`: CEM policy fitting.
- `robust_walker/metrics.py`: functional tracking and robustness metrics.
- `robust_walker/visualization.py`: single-run and side-by-side GIF animation helpers.
- `experiments/`: runnable protocols.

This is intentional. Sensor-Comm is the distributed sensory mechanism. CEM is only the optimizer used to fit policy parameters. Weak central control and dropout are experimental channels, not part of the Sensor-Comm definition.

## Controller Definitions

For limb `i`, all regimes share the same local/task/body baseline logit. The architectures differ in the additional three-parameter term:

```text
Capacity:
q_i <- q_i_local + W_i [a_i, a_i^2, tanh(2 a_i)]

Act-Comm:
q_i <- q_i_local + sum_{j != i} W_ij a_j

Sensor-Comm:
q_i <- q_i_local + sum_{j != i} W_ij s_j
```

Here `s_j` is the previous-step realized local force of limb `j`, normalized by the force scale. This is the crucial causal loop:

```text
a_j -> f_j -> s_j -> other limbs -> a_i(t+1)
```

Weak central control is modeled separately:

```text
q_i <- q_i + k_CC A_CC(t) e_goal side_i
```

## Current Verification

The historical files listed in the handoff were present, except `v4_1b_policies.npz` was not found in the Robust Walker source folder. The refactor therefore includes `experiments/01_train_policies.py` to regenerate that frozen policy file.

The current refactored run reproduced the reported V4.3 mean late path errors closely after regenerating policies:

| Disturbance condition | Reported Capacity | Reproduced Capacity | Reported Sensor-Comm | Reproduced Sensor-Comm |
| --- | ---: | ---: | ---: | ---: |
| CC dropout only | 0.598 | 0.575 | 0.637 | 0.616 |
| CC intermittent only | 0.714 | 0.689 | 0.683 | 0.658 |
| L4 failure only | 1.990 | 1.961 | 1.525 | 1.501 |
| L4 failure + CC dropout | 2.226 | 2.197 | 1.634 | 1.611 |
| L4 slip only | 0.890 | 0.859 | 0.706 | 0.681 |
| L4 slip + CC dropout | 1.014 | 0.990 | 0.771 | 0.751 |

Small numerical differences are expected because the original frozen policy archive was not present and was regenerated from the documented CEM protocol.

## Experiments

| Script | Purpose | Main outputs |
| --- | --- | --- |
| `01_train_policies.py` | Train matched Capacity, Act-Comm, and Sensor-Comm policies with vectorized CEM. | `policies/v4_1b_policies.npz`, `results/v4_1b_train.csv` |
| `02_limb_failure.py` | Evaluate held-out limb weakening and complete limb failure on ID and OOS tasks. | `results/limb_failure_results.csv`, `results/limb_failure_summary.csv` |
| `03_cc_dropout.py` | Reproduce the V4.3 weak-CC disturbance table: CC dropout, intermittent CC, L4 failure, L4 slip, and compound cases. | `results/cc_dropout_results.csv`, `results/cc_dropout_summary.csv` |
| `04_smoke_test.py` | Fast deterministic import/rollout/metric check. | Console pass/fail |
| `05_make_animations.py` | Generate representative side-by-side Capacity vs Sensor-Comm GIFs for CC dropout, L4 failure + CC dropout, and L4 slip + CC dropout. | `results/animations/*.gif`, `animation_cases.csv` |
| `06_cc_response_surface.py` | Sweep CC gain across normal, dropout, intermittent, delayed, noisy, and stuck CC modes under intact, L4 failure, and L4 slip conditions. | response-surface, minimum-gain, and retention CSVs |
| `07_make_spacious_summary_animation.py` | Generate the large, slideshow-readable V4.3d-style intermittent CC + L4 slip summary GIF. | `v4_3d_large_slow_intermittent_cc_l4_slip_spacious.gif` |

Run from this folder:

```bash
python -B experiments/04_smoke_test.py
python -B experiments/01_train_policies.py
python -B experiments/02_limb_failure.py
python -B experiments/03_cc_dropout.py
python -B experiments/05_make_animations.py
python -B experiments/06_cc_response_surface.py
python -B experiments/07_make_spacious_summary_animation.py
```

`05_make_animations.py` searches matched Capacity/Sensor-Comm seeds and tasks for representative cases with the largest Sensor-Comm late-error advantage, then writes:

- `results/animations/cc_dropout_only.gif`
- `results/animations/l4_failure_cc_dropout.gif`
- `results/animations/l4_slip_cc_dropout.gif`
- `results/animations/animation_cases.csv`

`06_cc_response_surface.py` runs the broader central-control experiment proposed for the talk: CC gain crossed with normal, dropout, intermittent, delayed, noisy, and stuck CC modes, under intact, L4 failure, and L4 slip conditions.

`07_make_spacious_summary_animation.py` creates a larger, slower V4.3d-style summary GIF for intermittent central control plus intermittent L4 slip:

- `results/animations/v4_3d_large_slow_intermittent_cc_l4_slip_spacious.gif`

For a fast local check of the pipeline:

```bash
python -B experiments/01_train_policies.py --seeds 9101 --iters 1 --pop 4 --elite 2
python -B experiments/02_limb_failure.py
```

## Next Scientific Tasks

- Add rollout logging for messages, local state, CC signal, and future functional state.
- Estimate channel-specific predictive information and functional predictive information.
- Extend and plot the CC strength by CC availability response surface.
- Define frozen held-out trajectory tests before additional tuning.
- Add a bottleneck/quantization layer for rate-distortion analysis.
