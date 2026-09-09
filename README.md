# DI-Walker: Distributed Intelligence and Failure Recovery

DI-Walker is a compact simulation testbed for studying when information exchanged among distributed components can improve robust function. The motivating question is:

> Can task-relevant sensory information shared among local components reduce dependence on a centralized controller and improve recovery when components, environmental coupling, or central supervision are disturbed?

The project uses a simple 2-D four-limbed embodied agent. The agent follows goal-conditioned trajectories while different controller architectures are tested under limb failure, intermittent slip, and degradation of a weak centralized correction signal.

The study is intentionally modest. It does not claim that distributed control always outperforms centralized control. A strong centralized feedback controller can solve many tracking problems. The interesting regime is lower central authority plus lateral sensory communication among components.

## Model

The simulated body has four limbs/actuators attached to a rigid 2-D body. The state contains position, heading, translational velocity, and angular velocity:

```text
state = [x, y, heading, vx, vy, omega]
```

At each time step, the task provides:

```text
v_d(t)      desired forward velocity
omega_d(t) desired yaw rate
```

The controller computes one command logit per limb:

```text
q_i
```

That logit is passed through a `tanh` nonlinearity to produce an instantaneous desired command. The actual actuator activation changes gradually:

```text
u_i = tanh(q_i)
a_i(t+1) = a_i(t) + dt * (u_i - a_i(t)) / tau
```

The realized force of limb `i` is proportional to its actuator activation:

```text
f_i = force_scale * a_i
```

When a limb is weakened, lost, or slipping, this realized force is reduced. The sensed local force is:

```text
s_i = f_i / force_scale
```

Thus `a_i` is the actuator state, while `s_i` is the realized local physical effect. This distinction is central to the experiment.

## Controller Architectures

All controller regimes share the same local/task/body feedback. The baseline logit for limb `i` includes velocity error, yaw-rate error, lateral body velocity, desired yaw rate, and the limb's own previous actuator state.

The architectures differ only in the additional three-parameter term.

### No-Force-Sensor

No-Force-Sensor is the minimal reference controller. It uses shared local/task/body feedback, but no realized-force sensor. The name is precise: the controller still observes body state and actuator state.

### Own-Sensor

Own-Sensor adds each limb's own realized-force sensor with one learned linear weight:

```text
q_i = q_i_local + w_i s_i
```

This isolates local realized-effect feedback from peer communication.

### Own-Kernel (legacy `capacity`)

Capacity is a matched-parameter no-peer-communication baseline. It uses each limb's own realized-force sensor, but no other limb's sensor:

```text
q_i = q_i_local + W_i [s_i, s_i^2, tanh(2s_i)]
```

This gives the baseline the same twelve architecture-specific parameters as Peer-Sensor, while providing local access to realized physical effect but no peer sensing. It remains available under the legacy identifier `capacity`.

### Peer-Sensor

Peer-Sensor gives each limb access to the other limbs' sensed realized forces:

```text
q_i = q_i_local + sum_{j != i} W_ij s_j
```

The causal loop is:

```text
a_j -> f_j -> s_j -> other limbs -> a_i(t+1)
```

This lets the controller respond to what another limb actually accomplished, not just what it attempted to do. The legacy identifier is `sensor_comm`.

### All-Linear

All-Linear gives every output limb the complete four-dimensional realized-force vector:

```text
q_i = q_i_local + sum_j W_ij s_j
```

It has sixteen sensor weights. It is an all-sensor linear controller, not the separate Central Control (CC) channel. The four conditions form an information-access design: No-Force-Sensor, Own-Sensor, Peer-Sensor, and All-Linear.

## Weak Central Control

Some experiments add a weak global correction signal. In this repository, `CC` means `central control`: a coarse global error channel, not a full high-bandwidth controller. CC observes target-relative path and heading error and adds a small left/right differential correction:

```text
q_i <- q_i + k_CC * A_CC(t) * e_goal * side_i
```

Here:

```text
k_CC     central-control gain
A_CC(t)  availability of central control at time t
e_goal   clipped combination of lateral path error and heading error
side_i   left/right limb sign
```

CC is not part of Sensor-Comm. It is a separate global channel used to test whether distributed sensory communication can reduce reliance on reliable centralized supervision.

The implemented CC error is:

```text
e_lat  = target displacement projected onto the body's lateral axis
e_head = wrapped(target_heading - body_heading)
e_goal = clip(0.65 * e_lat + 0.35 * e_head, -1, 1)
```

The availability term `A_CC(t)` implements central-control disturbances:

```text
always       A_CC(t) = 1
dropout      A_CC(t) = 1 before the dropout step, then 0
intermittent A_CC(t) = periodic on/off availability
delayed      CC uses an older error signal
noisy        CC receives additive noise
stuck        CC freezes at its value at the stuck step
```

## Disturbance Conditions

The experiments manipulate three kinds of robustness.

Component robustness:

```text
Limb weakening
Complete L4 failure
```

Environment/coupling robustness:

```text
Intermittent L4 slip
Out-of-sample trajectory shapes
```

Control-architecture robustness:

```text
CC dropout
Intermittent CC
Delayed CC
Noisy CC
Stuck CC
```

The hardest cases combine peripheral and central disturbance:

```text
L4 failure + CC dropout
L4 intermittent slip + intermittent CC
```

## Results

The main reported metric is mean late path error. Lower is better.

| Disturbance condition | Own-Kernel | Peer-Sensor | Peer advantage |
| --- | ---: | ---: | ---: |
| CC dropout only | 0.553 | 0.616 | -0.062 |
| CC intermittent only | 0.653 | 0.658 | -0.005 |
| L4 failure only | 1.758 | 1.501 | 0.257 |
| L4 failure + CC dropout | 1.961 | 1.611 | 0.350 |
| L4 slip only | 0.754 | 0.681 | 0.073 |
| L4 slip + CC dropout | 0.869 | 0.751 | 0.118 |

The interpretation is deliberately cautious.

Peer-Sensor is not uniformly better. In the CC-dropout-only condition, the own-sensor baselines can be competitive. This matters: the result does not support a blanket claim that distributed sensory communication solves every central-control failure.

The strongest Peer-Sensor advantage appears under limb failure, slip, and compound peripheral plus central disturbance. That supports a more precise hypothesis:

> Distributed sensory communication is most useful when surviving components need information about what other components are actually accomplishing, especially when centralized supervision is weak or unreliable.

This suggests a centralization-distribution tradeoff. Peer-Sensor may not beat an unrestricted centralized controller, but it can improve graceful degradation when local function and central supervision are both compromised.

## Repository Structure

```text
robust_walker/
  config.py          physical constants and body geometry
  tasks.py           goal-conditioned trajectory generators
  controllers.py     sensing architectures and weak CC
  faults.py          limb failure and intermittent slip disturbances
  simulation.py      canonical rollout implementation
  training.py        vectorized CEM policy fitting
  metrics.py         tracking and robustness metrics
  policies.py        frozen policy archive helpers
  visualization.py   GIF animation helpers

experiments/
  01_train_policies.py
  02_limb_failure.py
  03_cc_dropout.py
  04_smoke_test.py
  05_make_animations.py
  06_cc_response_surface.py
  07_make_spacious_summary_animation.py
  08_functional_predictive_information.py
  09_nonlinear_functional_predictive_information.py
  10_force_noise_sweep.py
  11_quantized_message_sweep.py
  12_plot_quantized_results.py
  13_theory_measurements.py
```

## Experiments

| Script | Purpose | Main outputs |
| --- | --- | --- |
| `01_train_policies.py` | Train No-Force-Sensor, Own-Sensor, Peer-Sensor, and All-Linear policies with vectorized CEM. | policy archive and training CSV |
| `02_limb_failure.py` | Evaluate held-out limb weakening and complete limb failure on ID and OOS tasks. | `results/limb_failure_results.csv`, `results/limb_failure_summary.csv` |
| `03_cc_dropout.py` | Evaluate weak-CC dropout, intermittent CC, L4 failure, L4 slip, and compound cases. | `results/cc_dropout_results.csv`, `results/cc_dropout_summary.csv` |
| `04_smoke_test.py` | Fast deterministic import/rollout/metric check. | Console pass/fail |
| `05_make_animations.py` | Generate representative side-by-side legacy Own-Kernel vs Peer-Sensor GIFs. | `results/animations/*.gif`, `animation_cases.csv` |
| `06_cc_response_surface.py` | Sweep CC gain across normal, dropout, intermittent, delayed, noisy, and stuck CC modes. | response-surface, minimum-gain, and retention CSVs |
| `07_make_spacious_summary_animation.py` | Generate a large, slideshow-readable intermittent-CC plus L4-slip summary GIF. | `v4_3d_large_slow_intermittent_cc_l4_slip_spacious.gif` |
| `08_functional_predictive_information.py` | Estimate global functional predictive-information rate and disturbance-induced information loss. | `results/functional_predictive_information.csv` |
| `09_nonlinear_functional_predictive_information.py` | Repeat the functional predictive-information estimate with a small nonlinear Gaussian predictor. | `results/nonlinear_functional_predictive_information_h10.csv` |
| `10_force_noise_sweep.py` | Test resilience to independent multiplicative realized-force noise on each limb. | `results/force_noise_sweep.csv` |
| `11_quantized_message_sweep.py` | Estimate a didactic rate-distortion curve by quantizing frozen sensor messages. | `results/quantized_message_sweep.csv` |
| `12_plot_quantized_results.py` | Plot the quantized-message rate-distortion results. | `results/quantized_message_rate_distortion.png` |
| `13_theory_measurements.py` | Estimate finite-time unstable expansion from numerical Jacobians and document directed channel rates. | `results/theory_measurements.csv`, `results/theory_measurements.png` |
| `15_functional_amortized_model.py` | Estimate message gain for future global cross-track error with an embodied MLP at selected horizons. | `results/functional_amortized_model.csv` |
| `16_plot_functional_amortized.py` | Plot completed functional-estimator CSV files without rerunning simulations. | `results/functional_amortized_horizon.png` |
| `17_history_functional_amortized.py` | Test sequential functional prediction using flattened causal histories of context and routed messages. | `results/history_functional_amortized.csv` |
| `18_history_estimator_audit.py` | Audit history estimation with validation early stopping, parameter-matched MLPs, expanded held-out routes/noise, and calibration diagnostics. | `results/history_estimator_audit.csv` |
| `20_history_scalar_calibration.py` | Repeat the focused history audit with more noise realizations and validation-fitted scalar predictive variance. | `results/history_scalar_calibration.csv` |
| `21_action_directed_information.py` | Estimate action-directed predictive information from realized-force messages to receiving-limb commands. | `results/action_directed_information.csv` |
| `22_plot_itpac_summary.py` | Create the integrated functional, action-directed, shuffled-message, and calibration figure. | `results/itpac_integrated_summary.png` |

`05_make_animations.py` accepts stronger disturbance settings for presentation examples. For example:

```bash
python -B experiments/05_make_animations.py \
  --tasks s_lr sine chirp \
  --cc-gain 0.10 --cc-drop-step 90 \
  --limb-start 50 --failure-strength 0.0 \
  --slip-strength 0.05 --slip-steps 18
```

The script searches the requested trajectories and keeps the representative case with the largest Sensor-Comm advantage for each animation scenario. Lower `--slip-strength`, longer `--slip-steps`, earlier `--limb-start`, and earlier `--cc-drop-step` create more severe conditions. The output GIFs and a CSV describing the selected seed, trajectory, and errors are written to `results/animations/`.

## Running

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a smoke test:

```bash
python -B experiments/04_smoke_test.py
```

Run the full pipeline:

```bash
python -B experiments/01_train_policies.py
python -B experiments/02_limb_failure.py
python -B experiments/03_cc_dropout.py
python -B experiments/05_make_animations.py
python -B experiments/06_cc_response_surface.py
python -B experiments/07_make_spacious_summary_animation.py
python -B experiments/08_functional_predictive_information.py
python -B experiments/10_force_noise_sweep.py
python -B experiments/11_quantized_message_sweep.py
python -B experiments/13_theory_measurements.py
python -B experiments/14_amortized_plant_model.py
MPLBACKEND=Agg python -B experiments/15_functional_amortized_model.py \
  --horizons 1 --epochs 60 \
  --out results/functional_amortized_h1.csv
MPLBACKEND=Agg python -B experiments/15_functional_amortized_model.py \
  --horizons 10 --epochs 60 \
  --out results/functional_amortized_h10.csv
MPLBACKEND=Agg python -B experiments/15_functional_amortized_model.py \
  --horizons 40 --epochs 60 \
  --out results/functional_amortized_h40.csv
MPLBACKEND=Agg python -B experiments/16_plot_functional_amortized.py \
  results/functional_amortized_h1.csv \
  results/functional_amortized_h10.csv \
  results/functional_amortized_h40.csv \
  --out results/functional_amortized_horizon.png
MPLBACKEND=Agg python -B experiments/17_history_functional_amortized.py \
  --histories 0 4 8 --horizons 1 10 40 --epochs 60 \
  --out results/history_functional_amortized.csv
MPLBACKEND=Agg python -B experiments/18_history_estimator_audit.py \
  --histories 0 4 8 --horizons 1 10 40 \
  --max-epochs 80 --patience 12 --noise-replicates 3 \
  --out results/history_estimator_audit.csv
MPLBACKEND=Agg python -B experiments/20_history_scalar_calibration.py \
  --histories 0 4 8 --horizons 1 40 \
  --max-epochs 80 --patience 12 \
  --train-noise-replicates 5 \
  --validation-noise-replicates 3 \
  --test-noise-replicates 8 \
  --out results/history_scalar_calibration.csv
```

For a faster check:

```bash
python -B experiments/01_train_policies.py --seeds 9101 --iters 1 --pop 4 --elite 2
python -B experiments/02_limb_failure.py
```

## Information-Theoretic Motivation

The theoretical report is being consolidated separately from this public repository. The public project documentation focuses on the reproducible simulation, controller definitions, and empirical results.

The current results are functional results: they measure tracking error and failure recovery. They do not by themselves prove that the communicated channels contain task-relevant predictive information.

`08_functional_predictive_information.py` estimates a global operational rate by comparing held-out prediction of future signed cross-track error with and without the architecture's routed force-sensor message vector. It reports `RF_bits_per_second` and the disturbance-induced change `LF_bits_per_second`. These are predictive-information estimates, not Shannon channel capacity. Small negative estimates are possible with finite held-out samples and should be interpreted as no measured incremental predictive gain, not as negative physical information.

`14_amortized_plant_model.py` is the pilot transition-model estimator. It trains matched probabilistic MLPs across routes and disturbance conditions, with and without the causal routed message, then evaluates held-out policy seeds or held-out route tasks. Use `--observation-mode full` for the analyst/debugging context or `--observation-mode embodied` for a receiver-centric local observation. Its output is an exploratory model-validation result, not yet a final Directed Information measurement.

The estimator now distinguishes limb slip, limb loss with the sensor retained, sensor failure without limb loss, and catastrophic limb loss with sensor loss. Use `--post-event` to score only transitions after the condition event and `--include-availability-mask` for a sensitivity analysis in which the estimator is explicitly told which sensor channels are unavailable.

The predictive-information analysis uses the causal sensor message available before each controller update. The realized force produced at step `t` becomes the sensor message available at step `t+1`; this is recorded separately from the post-action `force` signal in rollout logs. Existing predictive-information CSV files generated before this timing correction should be regenerated before using them as final results.

Policies are not selected by route. A policy archive stores one parameter vector per architecture and CEM seed, and the same frozen vector is evaluated on each task. Routes are represented by time-varying desired velocity and yaw-rate commands generated by `robust_walker/tasks.py`. ID tasks interpolate the training command family; OOS tasks use held-out route shapes or command parameters.

The next step is to estimate information-theoretic diagnostics from rollout logs, such as:

```text
PI_act(j -> i) = Delta LL(a_i,t+1 | local state, message_j,t)
PI_F(j -> i)   = Delta LL(Y_t+H | global/local state, message_j,t)
```

where `Y_t+H` is future task-relevant functional state.

This separates three claims:

```text
communication exists
communication predicts another component
communication predicts future function
```

A further extension is to introduce a communication bottleneck:

```text
S_t -> Z_t -> A_t+1
```

and study a rate-distortion-style curve:

```text
R = I(S; Z)
D = functional tracking loss
```

The deeper question is whether distributed sensory organization achieves robust function with less central supervision or lower communication capacity.

`17_history_functional_amortized.py` is the sequential extension. It compares a context-only predictor with a predictor that receives flattened causal windows of context and routed messages. The default run uses histories of 0, 4, and 8 steps and the same held-out route, policy-seed, disturbance, and force-noise protocol as experiment 15. The history result is currently an estimator diagnostic: larger message histories require a stronger training audit before negative gains can be interpreted as absence of directed information.

`18_history_estimator_audit.py` is the controlled estimator audit. It uses five training route families and three held-out route families, one validation policy seed, two held-out policy seeds, three held-out force-noise realizations, validation-based early stopping, approximately parameter-matched context-only and message MLPs, and 50/80/95 percent Gaussian interval coverage diagnostics. Its summary is in `results/history_estimator_audit_summary.csv`.

`20_history_scalar_calibration.py` is the stable higher-data follow-up. It keeps the mean MLP unchanged, increases the independent force-noise realizations to 5 training, 3 validation, and 8 test realizations, and fits one scalar residual variance from validation data. The directly learned heteroscedastic-variance pilot in `19_history_calibrated_audit.py` was not used for conclusions because it was overconfident and numerically unstable under route shifts. The scalar-calibration summary is in `results/history_scalar_calibration_summary.csv`.

`21_action_directed_information.py` estimates the IT-PAC-relevant action channel. Its target is the receiving limb's post-policy command `control[i,t]`, while the input message is the realized-force sensor available before that decision. It compares context-only and message-conditioned predictors, using the same histories, validation early stopping, held-out route/policy split, and force-noise replicates as the calibrated audit. Its gain is reported in bits per action. This is distinct from the global functional estimator: it measures whether messages influence action selection, not whether they directly predict future tracking error.

`22_plot_itpac_summary.py` combines the completed functional and action-directed CSV files into a four-panel figure. It does not rerun simulations.

```bash
MPLBACKEND=Agg python -B experiments/22_plot_itpac_summary.py \
  --out results/itpac_integrated_summary.png
```

The detailed internal account of the current simulation and theory is in
`reports/di-walker-detailed-simulation-theory-report.tex` and its locally compiled PDF. The report is intentionally excluded from the public repository while the theoretical drafts are being consolidated.
