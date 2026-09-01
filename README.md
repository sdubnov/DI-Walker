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

### Capacity

Capacity is a matched-parameter no-peer-communication baseline. It uses each limb's own realized-force sensor, but no other limb's sensor:

```text
q_i = q_i_local + W_i [s_i, s_i^2, tanh(2s_i)]
```

This gives the baseline the same number of additional parameters as the communication architectures, and it also gives the baseline local access to realized physical effect. What it does not provide is peer sensing: limb `i` cannot directly read `s_j` for `j != i`. This controls for the possibility that Sensor-Comm performs better merely because it has more parameters or because it uses sensors at all.

### Act-Comm

Act-Comm gives each limb access to the other limbs' actuator activations:

```text
q_i = q_i_local + sum_{j != i} W_ij a_j
```

This is cross-limb communication, but it communicates intended or internal actuator state rather than realized physical effect.

Act-Comm is included as a diagnostic comparison. The main failure-recovery claim does not rely on it: if a limb is slipping or failed, its activation `a_j` can still indicate that it is trying to push, while its sensor `s_j` reports that little or no force was actually produced.

### Sensor-Comm

Sensor-Comm gives each limb access to the other limbs' sensed realized forces:

```text
q_i = q_i_local + sum_{j != i} W_ij s_j
```

The causal loop is:

```text
a_j -> f_j -> s_j -> other limbs -> a_i(t+1)
```

This lets the controller respond to what another limb actually accomplished, not just what it attempted to do. That is why Sensor-Comm is the main distributed-information architecture in this repository.

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

| Disturbance condition | Capacity | Sensor-Comm | Sensor advantage |
| --- | ---: | ---: | ---: |
| CC dropout only | 0.553 | 0.616 | -0.062 |
| CC intermittent only | 0.653 | 0.658 | -0.005 |
| L4 failure only | 1.758 | 1.501 | 0.257 |
| L4 failure + CC dropout | 1.961 | 1.611 | 0.350 |
| L4 slip only | 0.754 | 0.681 | 0.073 |
| L4 slip + CC dropout | 0.869 | 0.751 | 0.118 |

The interpretation is deliberately cautious.

Sensor-Comm is not uniformly better. In the CC-dropout-only condition, Capacity is slightly better on average. This matters: the result does not support a blanket claim that distributed sensory communication solves every central-control failure.

The strongest Sensor-Comm advantage appears under limb failure, slip, and compound peripheral plus central disturbance. That supports a more precise hypothesis:

> Distributed sensory communication is most useful when surviving components need information about what other components are actually accomplishing, especially when centralized supervision is weak or unreliable.

This suggests a centralization-distribution tradeoff. Sensor-Comm may not beat an unrestricted centralized controller, but it can improve graceful degradation when local function and central supervision are both compromised.

## Repository Structure

```text
robust_walker/
  config.py          physical constants and body geometry
  tasks.py           goal-conditioned trajectory generators
  controllers.py     Capacity, Act-Comm, Sensor-Comm, and weak CC
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
```

## Experiments

| Script | Purpose | Main outputs |
| --- | --- | --- |
| `01_train_policies.py` | Train matched Capacity, Act-Comm, and Sensor-Comm policies with vectorized CEM. | `policies/v4_1b_policies.npz`, `results/v4_1b_train.csv` |
| `02_limb_failure.py` | Evaluate held-out limb weakening and complete limb failure on ID and OOS tasks. | `results/limb_failure_results.csv`, `results/limb_failure_summary.csv` |
| `03_cc_dropout.py` | Evaluate weak-CC dropout, intermittent CC, L4 failure, L4 slip, and compound cases. | `results/cc_dropout_results.csv`, `results/cc_dropout_summary.csv` |
| `04_smoke_test.py` | Fast deterministic import/rollout/metric check. | Console pass/fail |
| `05_make_animations.py` | Generate representative side-by-side Capacity vs Sensor-Comm GIFs. | `results/animations/*.gif`, `animation_cases.csv` |
| `06_cc_response_surface.py` | Sweep CC gain across normal, dropout, intermittent, delayed, noisy, and stuck CC modes. | response-surface, minimum-gain, and retention CSVs |
| `07_make_spacious_summary_animation.py` | Generate a large, slideshow-readable intermittent-CC plus L4-slip summary GIF. | `v4_3d_large_slow_intermittent_cc_l4_slip_spacious.gif` |

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
```

For a faster check:

```bash
python -B experiments/01_train_policies.py --seeds 9101 --iters 1 --pop 4 --elite 2
python -B experiments/02_limb_failure.py
```

## Information-Theoretic Motivation

The current results are functional results: they measure tracking error and failure recovery. They do not by themselves prove that the communicated channels contain task-relevant predictive information.

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
