# Distributed Intelligence, Disturbance, and Robust Control

Prepared for the ELLIIT symposium work session on October 23.

The working hypothesis is not simply that more directed information creates robustness. Earlier experiments showed that rewarding dependency can make policies stiff. The more precise claim is that task-relevant predictive information among components may support distributed coordination and reduce dependence on reliable centralized supervision under disturbance or failure.

The most important empirical mechanism is Sensor-Comm:

```text
q_i = q_i_local + sum_{j != i} W_ij s_j
```

where `s_j` is another limb's realized force, available on the next control step. This differs from Act-Comm, which communicates actuator activation rather than realized local physical consequence.

The core comparison should therefore remain:

- matched-capacity no-peer-communication controller
- actuator communication
- realized-force sensory communication

The strongest current claims are functional, not yet information-theoretic. Lower excess loss under limb loss/slip indicates better robustness, but it does not by itself establish predictive information, directed information, or transfer entropy in the communicated channel.

The next theory-aligned measurements should estimate:

```text
PI_act(j -> i) = Delta LL(a_i,t+1 | local state, message_j,t)
PI_F(j -> i)   = Delta LL(Y_t+H | global/local state, message_j,t)
```

where `Y_t+H` is future task-relevant functional state. This separates communication, component prediction, and future functional prediction.

The next experimental extension is a bottleneck:

```text
S_t -> Z_t -> A_t+1
```

with variable capacity through dimensional restriction, quantization, bits/channel, or learned compression. The rate-distortion question is whether Sensor-Comm achieves lower functional distortion at a given information rate than matched Capacity control.
