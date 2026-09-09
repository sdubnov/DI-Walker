# DI-Walker Lecture Notes: Working Outline

## 1. Lecture question

Can information exchanged among distributed components help a physical system continue its task when a limb, the environment, or centralized supervision fails?

The careful version of the claim is not that distributed control always beats centralized control. The current study asks whether lateral sensory communication can reduce dependence on reliable global supervision and improve graceful degradation.

## 2. The demonstration system

DI-Walker is a two-dimensional four-limb embodied agent following goal-conditioned routes.

The plant state is:

```text
[x, y, heading, vx, vy, omega]
```

The task supplies desired forward velocity and desired yaw rate. The controller produces one command logit per limb. The logit is passed through `tanh`, and the actuator state changes with a first-order time constant. Realized limb force is then fed back as a sensed local effect.

Relevant implementation:

- `robust_walker/config.py`: physical constants and geometry.
- `robust_walker/tasks.py`: route and command generation.
- `robust_walker/simulation.py`: causal rollout timing and disturbance handling.
- `robust_walker/controllers.py`: controller architectures.

The critical timing is:

```text
message[t] -> controller command[t] -> plant force[t]
            -> sensed realized effect[t+1]
```

## 3. The controller comparison

The public names are:

- **No-Force-Sensor**: body, task, and actuator feedback, but no realized-force sensor.
- **Own-Sensor**: each limb receives its own realized-force sensor.
- **Peer-Sensor**: each limb receives the other limbs' realized-force sensors.
- **All-Linear**: each limb receives the complete four-dimensional force-sensor vector.

The older `capacity` identifier refers to the legacy Own-Kernel baseline. It should not be used in lecture slides as an information-theoretic capacity claim.

The sensor distinction is physical rather than merely architectural:

```text
activation a_i -> realized force f_i -> sensor s_i
```

An actuator command says what a limb was asked to do. A realized-force sensor says what the limb actually accomplished. Under slip or weakening, these can differ. That is why peer realized-effect communication is the relevant distributed channel for failure recovery.

## 4. Central Control (CC)

`CC` means weak **central control**, not centralization of the entire policy. It observes a scalar clipped combination of target-relative lateral and heading error and sends a small signed correction to the limbs.

The correction is scaled by:

- `k_CC`: central-control gain;
- `A_CC(t)`: availability, which can drop out, become intermittent, delayed, noisy, or stuck;
- `side_i`: the fixed left/right sign of limb `i`.

CC is a separate global information channel. It is not the same as Peer-Sensor communication.

Relevant implementation:

- `robust_walker/controllers.py`: CC correction.
- `robust_walker/faults.py`: CC and limb disturbances.
- `experiments/03_cc_dropout.py`, `06_cc_response_surface.py`: CC experiments.

## 5. Training and testing

Policies are learned with curriculum-trained CEM. A policy archive stores parameter vectors, not fixed action sequences. The same frozen policy is evaluated on multiple routes; a route is not selected by loading a route-specific policy.

Training routes and held-out routes are generated from different task families in `robust_walker/tasks.py`. Disturbances are introduced during evaluation unless an explicitly noise-trained archive is used.

Relevant implementation:

- `robust_walker/training.py`: CEM and policy training.
- `robust_walker/policies.py`: policy archive loading and evaluation.
- `experiments/01_train_policies.py`: standard training.
- `experiments/02_limb_failure.py`, `05_make_animations.py`, `07_make_spacious_summary_animation.py`: physical demonstrations.

## 6. First empirical result: functional robustness

The primary physical metric is late tracking error. Lower error means that the body remains closer to its target route after a disturbance.

The important pattern is conditional, not universal:

- Peer-Sensor is not uniformly better under CC dropout alone.
- Peer-Sensor is more favorable in several limb-failure, slip, and compound limb-plus-CC conditions.

This supports the restrained interpretation:

> Lateral realized-effect communication can improve graceful degradation when the system needs information about what another limb is actually accomplishing, particularly when central supervision is also weak.

The animations are didactic demonstrations of selected trajectories, not population-level statistical evidence. Population summaries should be used for claims.

## 7. Directed-information motivation

The causal channel is:

```text
peer realized effect -> receiving-limb action -> future body state
```

Directed information is relevant because the message is available before the receiving action and can influence later states. The essential question is not whether the sensor values are statistically dependent, but whether their time-ordered content improves prediction or control of future variables.

The current measurements separate two targets.

### 7.1 Action-directed gain

`experiments/21_action_directed_information.py` predicts the receiving limb's post-policy command with and without the causal message. Its operational gain is:

```text
message gain = NLL(context-only) - NLL(context + message)
```

divided by `ln(2)` to express the difference in bits per action decision.

The shuffled-message control preserves message values while breaking their route/time alignment. If true messages outperform shuffled messages, the result is evidence that temporally relevant message content helps action prediction.

Current interpretation: Peer-Sensor shows a promising action-prediction gain under compound failure. This is evidence for action-directed predictive influence, not yet a calibrated total communication rate or proof of the data-rate theorem.

### 7.2 Functional gain

`experiments/15_functional_amortized_model.py`, `17_history_functional_amortized.py`, and `20_history_scalar_calibration.py` predict future global tracking error. The target is a future functional variable `Y[t+H]`, where `H` is the plant/task horizon.

```text
functional gain(H) = NLL(context-only) - NLL(context + message)
```

Panel A of the integrated figure varies `H`. It asks whether the message predicts future task function.

The current global estimator gives a stable positive long-horizon Own-Sensor result but does not establish a reliable positive Peer-Sensor global gain. This is an estimator result, not evidence that peer communication is physically useless.

## 8. How to read the integrated figure

`results/itpac_integrated_summary.png` is generated by `experiments/22_plot_itpac_summary.py`.

- **Panel A:** functional gain versus future horizon `H`.
- **Panel B:** action gain versus input history length `k`. This is not the same as future horizon `H`.
- **Panel C:** action-prediction NLL for context-only, true-message, and shuffled-message inputs in the compound condition. Lower NLL is better, and negative NLL values are possible for continuous Gaussian densities.
- **Panel D:** empirical interval coverage. It diagnoses whether the scalar Gaussian residual model is calibrated; it is not a measure of plant-action variance.

Panel C should primarily be read within each architecture. The meaningful comparison is the reduction from context-only to true message, and whether true message beats shuffled message. Raw Own-versus-Peer NLL levels can be affected by different action distributions and separately calibrated residual variances.

## 9. What is established and what is not

### Established by the current code and experiments

- The simulator has an explicit causal realized-force message path.
- Policies are frozen during the main disturbance tests.
- The physical system can be animated under limb and CC disturbances.
- Peer messages can improve prediction of receiving-limb actions in selected failure conditions.
- Functional tracking error and predictive gains are distinct measurements.
- The current protocol does not show a universal Peer-Sensor robustness advantage.

### Not established yet

- A formal directed-information rate in the Shannon/data-rate-theorem sense.
- A proof that peer predictive information causes the observed functional recovery advantage.
- A reliable positive Peer-Sensor gain for the current global future-error estimator.
- A calibrated bit rate for continuous messages.
- A Lyapunov stability theorem for the nonlinear closed-loop simulator.

## 10. Minimum next work before final lecture claims

1. Run inference-time message ablation with the frozen policies: true, zeroed, and shuffled messages.
2. Pair action-directed gain with the resulting change in recovery error.
3. Report confidence intervals over policy seeds, routes, and disturbance realizations.
4. Keep the language “action-directed predictive information” or “IT-PAC-related operational measure” unless a stronger estimator and causal test justify a stronger claim.
5. Present the formal data-rate and Lyapunov relations as theoretical motivation and finite-sample diagnostics, not as proofs supplied by this simulation.

## 11. Suggested lecture arc

1. Start with the animated failure: normal operation, limb disturbance, and loss of CC.
2. Explain the physical distinction between command, realized force, and sensed realized effect.
3. Compare No-Force-Sensor, Own-Sensor, Peer-Sensor, and All-Linear.
4. Show the recovery-error result and emphasize the compound-disturbance pattern.
5. Introduce the causal message-to-action-to-function chain.
6. Use Panel C to explain true versus shuffled messages.
7. Use Panel A to explain functional prediction across future horizons.
8. Use Panel D to explain estimator calibration and why the current bits are provisional.
9. Close with the distinction between predictive influence, functional utility, and communication rate.
