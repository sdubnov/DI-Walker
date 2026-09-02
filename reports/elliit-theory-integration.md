# Directed Information, Robust Control, and Distributed Intelligence

## Purpose

This note integrates the ELLIIT theoretical reports with the background readings and the Directed Information / SymTE work. It is intended to provide the theoretical spine for the DI-Walker quadruped study and a disciplined basis for the symposium presentation.

The central question is:

> When does information exchanged among distributed components become functionally useful, and can it reduce dependence on centralized supervision under disturbances and failures?

The answer should not be stated as “more information produces more robustness.” The simulation and the theory support a narrower claim: information is useful when it is causally available, predictive of relevant future states, and connected to the control actions needed to preserve function.

## 1. The Reports Are Successive Drafts

The files in `Reports/` are not independent analyses with equal status. They form an evolving sequence.

### Report 1: broad theoretical synthesis

`Generalized_Data_Rate_Inequality (1).pdf` combines several established ideas:

- Pesin entropy and positive Lyapunov exponents as a rate of dynamical information production.
- The linear Data Rate Theorem as a lower bound on feedback communication for stabilization.
- The Bode sensitivity integral as a frequency-domain tradeoff in disturbance rejection.
- A proposed decomposition of information flow in distributed systems.
- A proposed centralized versus hybrid distributed architecture for the quadruped.

Its value is architectural and motivational. Its weakness is that it moves too quickly from analogies and established linear results to a claimed distributed nonlinear principle. The code fragment for a one-step Gaussian KL divergence is also not a measurement implemented by the current DI-Walker experiments.

### Report 2: latent data-rate principle

`Generalized_Data_Rate_Inequality (2).pdf` attempts a formal generalization. It starts with a geometric volume-expansion argument for the linear Data Rate Theorem, then replaces:

- unstable eigenvalue growth with latent entropy generation,
- fixed channel capacity with Directed Information,
- full physical state with a learned latent manifold.

The conceptual direction is promising, but the central "if and only if" claim is too strong. A necessary data-rate condition for a restricted class of networked control systems is not automatically a sufficient condition for a nonlinear learned controller with faults. The report also treats latent entropy generation as if it were a uniquely defined quantity independent of representation, estimator, mode switching, and noise model.

### Reports 3 and 4: corrected data-rate derivation

`Generalized_Data_Rate_Inequality (3).pdf` and `Generalized_Data_Rate_Inequality_REVISED.pdf` are the most mature versions. They make three important corrections:

1. They use singular values of a product of non-commuting Jacobians, rather than instantaneous eigenvalues.
2. They restrict the expansion term to unstable Oseledets directions, rather than using the full-space log determinant.
3. They state that the current robot experiment demonstrates a fault-sensitive sensory pathway and functional tracking differences, but does not yet estimate fault-conditioned predictive information or local Lyapunov rates.

The revised report should therefore be treated as the theoretical baseline. Report 3 is useful as a parallel exposition, but the revised report has the clearest separation between the theorem, the operational information diagnostic, and the empirical limitations.

## 2. Main Discrepancies and Corrections

### Full determinant versus unstable expansion

Earlier derivations use a quantity such as `log |det J|` or a volume expansion over the complete state space. This is not generally the right data-rate quantity for stabilization. Stable directions can contract while unstable directions expand. Full-space contraction can therefore hide an unstable direction that still requires feedback information.

For a smooth system with an invariant measure satisfying the hypotheses of Pesin's formula, the relevant asymptotic rate is the sum of positive Lyapunov exponents:

```text
h_nu(f) = integral over M of sum_{lambda_i(z) > 0} lambda_i(z) d nu(z)
```

For a finite trajectory, the practical analogue is a finite-time unstable expansion rate based on singular values of a Jacobian product:

```text
R_crit_hat(t,H) = [1 / (H log 2)] sum_{sigma_k(M_t,H) > 1} log sigma_k(M_t,H)
```

This is a local diagnostic, not yet a computed result of DI-Walker.

### Topological entropy versus metric entropy

The reports sometimes use topological entropy, metric entropy, and Pesin entropy as if they were interchangeable. They are not. Metric entropy depends on an invariant measure; topological entropy is a property of the map and bounds metric entropy through the variational principle. Equality requires additional assumptions. The report should use “positive Lyapunov expansion rate” for the local control argument and reserve “metric entropy” for the measure-dependent statement.

### Directed Information versus Transfer Entropy

Directed Information in a feedback system is normally a causal, sequence-level quantity. A standard form is:

```text
I(X^T -> Y^T) = sum_t I(X^t ; Y_t | Y^(t-1))
```

Transfer Entropy usually uses the source past rather than the source present:

```text
TE(X -> Y) = I(X^(t-1) ; Y_t | Y^(t-1))
```

Under strictly causal communication and suitable timing assumptions, the distinction can be aligned. It cannot simply be ignored. The current robot message is available with one-step delay, so the experiment is naturally compatible with a predictive, past-to-next-step formulation. The exact conditioning set must nevertheless be stated for every estimator.

### “DI outpaces entropy” is not an established nonlinear theorem here

The corrected report derives a necessary inequality under explicit smoothness, invariant-measure, and causal-channel assumptions. It does not prove that a learned nonlinear controller is stabilizable if and only if an empirically estimated DI exceeds a latent entropy rate. That stronger result would require a new theorem specifying:

- the plant and fault modes;
- the coding and decoding architecture;
- the meaning of stabilization or bounded tracking;
- the noise and observability assumptions;
- the relationship between the information decomposition and the total information available to control.

The symposium report should call the nonlinear statement a proposed principle or conjecture, not a proved theorem.

### The robot report has implementation discrepancies

The reports describe a representative simulation with `dt = 0.01` in places, while the public DI-Walker code uses `DT = 0.05` and `T = 240`. Older text also describes the Capacity baseline using its own activation, whereas the current public code uses its own realized-force sensor. The public code is the source of truth for current numerical results.

The current simulation is deterministic conditional on task, policy seed, and disturbance schedule. It does not automatically provide an invariant ensemble, a stochastic channel, or a measured communication bit rate. Consequently, its existing tracking and recovery results should not be presented as a direct empirical verification of the Data Rate Theorem.

## 3. Relation to Information Dynamics, DI, and SymTE

### Information Rate and predictive information

Information Rate and Music Information Dynamics measure how much uncertainty about the future is reduced by the past. In a generic process `X`, a predictive-information increment can be represented as:

```text
PI_X(t) = I(X^(t-1) ; X_t)
```

or, operationally, as the improvement in predictive log likelihood when a past context is supplied. This is a temporal measure: it concerns prediction, not static dependence.

### Directed information as the multi-track extension

For interacting tracks or components, one asks whether the past of one process improves prediction of another process after conditioning on the target's own past:

```text
TE(j -> i) = I(S_j^(t-1) ; A_i,t | A_i^(t-1), C_i,t)
```

Here `C_i,t` contains the common task, body, and central-control variables already available to component `i`. This conditioning is essential. Without it, shared task drive or common body motion can be mistaken for communication.

For a closed-loop system, a sequence-level Directed Information quantity is preferable when the present source value can influence the present target value. In DI-Walker, the peer force message is delayed by the simulator update, so the one-step predictive transfer-entropy form is a natural operational estimator.

### SymTE and total multi-track coordination

SymTE was developed for two interacting musical processes and combines the two directional transfer terms:

```text
SymTE(X,Y) = I(Y_t ; X^(t-1) | Y^(t-1))
           + I(X_t ; Y^(t-1) | X^(t-1))
```

The later total-information-flow work extends the intuition to multitrack sequences by comparing predictive surprisal for each track with surprisal for the joint sequence. A generative model supplies an operational estimate of conditional entropy through next-event log likelihood.

The useful generalization for the quadruped is not to copy the musical score literally. It is to retain the hierarchy:

```text
predictive information within a component
    -> directional information between components
    -> joint multi-component information
    -> information about future task function
```

The last arrow is the crucial addition. High mutual or directed information can represent synchronized failure, common drive, or redundant behavior without improving recovery. The relevant quantity is therefore fault-conditioned and task-conditioned predictive information.

## 4. A Unified Closed-Loop Model for DI-Walker

Let the physical/controller state be:

```text
z_t = [body_state_t, actuator_state_t]
```

For a fixed task reference `r_t` and disturbance mode `m_t`, the simulator defines:

```text
z_(t+1) = F_m(z_t, r_t)
```

For limb `i`:

```text
q_i,t  = local_i,t + communication_i,t + CC_i,t
u_i,t  = tanh(q_i,t)
a_i,t+1 = a_i,t + DT * (u_i,t - a_i,t) / TAU
f_i,t  = FORCE_SCALE * a_i,t * e_i,t
s_i,t  = f_i,t / FORCE_SCALE
```

The symbols have distinct roles:

- `q_i`: pre-activation logit, before saturation.
- `u_i`: instantaneous desired command after `tanh`.
- `a_i`: smoothed actuator state.
- `f_i`: realized physical force.
- `e_i`: disturbance-dependent retained-strength multiplier.
- `s_i`: normalized realized-force sensor.

The information graph is:

```text
task and body state -> local controller -> q_i -> u_i -> a_i -> f_i -> s_i
                                               ^                         |
                                               |------ peer sensors -----|
```

The critical Sensor-Comm distinction is that a peer receives `s_j`, not `a_j`. Under a limb failure, `a_j` can remain large because the controller is still attempting to act, while `s_j` falls because the physical effect has been lost. Sensor-Comm therefore exposes a fault-sensitive consequence rather than only an intention or command.

### Capacity is not Shannon channel capacity

In the public code, `Capacity` is a name inherited from the experimental comparison. It is a matched-parameter no-peer-sensing baseline:

```text
q_i = q_i_local + W_i [s_i, s_i^2, tanh(2 s_i)]
```

The word “capacity” here refers to matched controller capacity or representational capacity, not measured communication capacity in bits per step. This distinction should be made explicit in the talk. A future bottleneck experiment may use `C` or `R` for an actual channel capacity and reserve `Capacity` for the baseline only if the naming is unavoidable.

### Central control (CC)

`CC` means central control. It is a weak global error channel, not an unrestricted centralized policy. The implemented correction is:

```text
q_i <- q_i + k_CC * A_CC(t) * e_goal(t) * side_i
```

`e_goal` combines lateral target error and heading error. `A_CC(t)` models availability, including dropout, intermittent, delayed, noisy, and stuck modes. CC and Sensor-Comm are separate information channels: CC conveys coarse global task error, while Sensor-Comm conveys peer realized physical effects.

## 5. How the Theory Explains the Simulation

The public experiments compare local-sensor Capacity with peer-sensor Sensor-Comm under intact operation, limb failure, intermittent slip, out-of-sample trajectories, and CC disturbance. The current functional metric is late path error and its change relative to intact operation.

The theoretical interpretation should be layered.

### Layer 1: functional redundancy

When one limb loses force, the remaining limbs can use body-level feedback to compensate. This is ordinary feedback robustness and does not by itself demonstrate distributed intelligence.

### Layer 2: fault-sensitive causal information

Sensor-Comm gives each limb a delayed observation of what other limbs actually produced. A force discrepancy caused by a failure enters the peer message and can alter the next control action. This establishes a causal pathway that is sensitive to the physical consequence of a disturbance.

### Layer 3: predictive coordination

The stronger information-theoretic claim is that peer sensory history improves prediction of another limb's next action or of future body/task state, conditional on local state and common inputs. This has not yet been measured in the current code and should be presented as the next test.

### Layer 4: robust control under information loss

CC supplies global error information. A CC failure removes one route to task correction. A limb failure removes one physical actuation route. The compound condition removes both a physical resource and a supervisory information resource. Sensor-Comm can be advantageous there because peer sensing preserves a lateral information pathway even when central supervision is unavailable.

This is a distributed robust-control interpretation, not a claim that decentralization dominates centralized control. A sufficiently powerful centralized controller with reliable global sensing may perform better. The interesting claim concerns graceful degradation under partial observability, communication loss, actuator failure, and limited central authority.

## 6. Recommended Information-Theoretic Analysis

The following measurements would turn the current functional study into a direct test of the theoretical story.

### 6.1 Component predictive information

For each active limb `i`, estimate the incremental predictive information supplied by peer `j`:

```text
PI_act(j -> i)
  = LL[a_i,t+1 | local_i,t, peer_j,t]
  - LL[a_i,t+1 | local_i,t]
```

Use the same model class, held-out data, and cross-validation for both predictors. This measures whether peer sensing predicts another component's action, not whether it improves the mission.

### 6.2 Functional predictive information

Define a future task-relevant variable `Y_(t+H)`, such as future lateral error, heading error, body velocity, or a discretized recovery-success event:

```text
PI_F(j -> function)
  = LL[Y_(t+H) | context_t, peer_j,t]
  - LL[Y_(t+H) | context_t]
```

The context must include variables available to the receiving limb or controller. Estimate this separately for intact, limb-failure, slip, and CC-dropout modes. The predicted quantity should be fixed before analysis to avoid selecting an outcome that favors Sensor-Comm.

### 6.3 Fault-conditioned information gain

The most relevant comparison is not simply `I(peer; action)`. It is the change induced by the disturbance:

```text
Delta_PI_F = PI_F under failure - PI_F intact
```

The hypothesis is that peer sensory information becomes more functionally valuable when the physical relation between command and effect changes. This directly tests the distinction between command communication and realized-effect communication that motivated removal of Act-Comm from the public comparison.

### 6.4 Multi-limb information decomposition

For the set of peer messages `S_-i`, estimate:

```text
I(S_-i ; Y_future | context)
```

Then use a source-decomposition method, with appropriate caveats, to separate unique, redundant, and synergistic contributions. A useful empirical question is whether recovery depends on one critical peer edge or on distributed redundancy across several limbs.

SymTE provides a useful two-track intuition, but a four-limb system requires a multivariate definition and careful treatment of redundancy and synergy. Pairwise sums are not automatically equal to total network information.

### 6.5 Local stability and data-rate diagnostics

For each fixed disturbance mode, estimate finite-time Jacobian products of the actual simulator map, then compute positive singular-value growth. Compare this local expansion diagnostic with the information available to the relevant controller. This would be an empirical connection to the corrected data-rate inequality, but it should be described as a local finite-horizon comparison rather than a proof of the inequality.

## 7. Robust-Control Interpretation

The classical networked-control literature supports three ideas relevant here.

First, stabilization through communication constraints is governed by the tension between unstable state expansion and the rate, delay, reliability, and timing of feedback. The Data Rate Theorem is the clean linear reference point.

Second, decentralized estimation and control depend on which node observes which part of the state and how information is routed. A network can preserve useful local estimates even when a central route is delayed or unavailable.

Third, information has value only relative to a control objective. The same message can be informative statistically but irrelevant to stabilizing the task. This is why the DI-Walker extension should use functional predictive information and not only raw transfer entropy.

The Bode sensitivity integral is best used as a qualitative control-theoretic analogy in this project. It expresses a frequency-domain tradeoff for appropriate linear feedback systems; it does not directly quantify the GIFs or establish that Sensor-Comm must improve their path error.

## 8. Distributed Intelligence: A Precise Working Definition

For this project, distributed intelligence should mean:

> The ability of multiple partially informed components to use causally available, task-relevant information about one another's realized effects to preserve a shared function when local components or centralized supervision are unreliable.

This definition has four testable parts:

1. **Partial information:** no limb has the complete global state.
2. **Causal exchange:** peer information arrives with an explicit timing relation.
3. **Functional relevance:** the information predicts or improves a future task state.
4. **Graceful degradation:** function is retained under component, environment, or central-control disturbance.

It excludes several weaker notions:

- parameter count alone;
- statistical correlation caused by shared task drive;
- high activity or high mutual information without improved prediction;
- centralized feedback relabeled as distributed control;
- recovery that requires retraining when the claim concerns intrinsic robustness.

## 9. Claims Appropriate for the Symposium

The strongest defensible presentation is:

1. The classical Data Rate Theorem motivates comparing dynamical uncertainty generation with causal information delivery.
2. Predictive information and Music Information Dynamics measure temporal information within a process; Directed Information and Transfer Entropy extend this to directional interaction; SymTE extends the intuition to bidirectional multi-track coordination.
3. The quadruped instantiates a distributed multi-track closed loop in which limbs exchange delayed realized-force measurements.
4. The current functional results show that Sensor-Comm is most useful under limb failure, slip, and compound limb plus CC disturbances. It is not uniformly better under CC dropout alone.
5. The current code establishes the existence of a fault-sensitive causal pathway and a functional difference. It does not yet establish a quantitative DI-to-robustness law.
6. The next experiment should measure fault-conditioned functional predictive information and compare it with recovery error and finite-time unstable expansion.

The resulting thesis is:

> Distributed intelligence is not simply more information or more communication. It is the organization of limited, causally routed, task-relevant predictive information so that a system can preserve function when its components and supervisory channels are imperfect.

## References and Source Material

- ELLIIT local reports: `/Users/sdubnov/Documents/Research/Talks/ELLIIT/Reports/`
- ELLIIT background readings: `/Users/sdubnov/Documents/Research/Talks/ELLIIT/To Read/`
- Dubnov, Gokul, and Assayag, “Switching Machine Improvisation Models by Latent Transfer Entropy Criteria,” [Physical Sciences Forum](https://www.mdpi.com/2673-9984/5/1/49).
- Gokul, Francis, and Dubnov, “Evaluating Co-Creativity using Total Information Flow,” [arXiv:2402.06810](https://arxiv.org/abs/2402.06810).
- Franceschetti, Khojasteh, and Win, “The Many Facets of Information in Networked Estimation and Control,” [Annual Review of Control, Robotics, and Autonomous Systems](https://doi.org/10.1146/annurev-control-042820-010811).
- DI-Walker implementation: `robust_walker/`, especially `controllers.py`, `simulation.py`, and `faults.py`.
