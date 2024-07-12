# Compared Method 2

This is an implementation of the model in paper "Reinforcement Learning with a Network of Spiking Agents" ([Link to paper](https://openreview.net/forum?id=B1eU47t8Ir)).

This implementation is based on code from [spiking-agent-RL/spiking_agent_cartpole.py at master · asneha213/spiking-agent-RL · GitHub](https://github.com/asneha213/spiking-agent-RL/blob/master/spiking_agent_cartpole.py).

Some changes are listed below:

* Environment changed to GYMIP.

* In GYMIP, the state variables clipped to [-1, 1]. The input encoding (limit) is changed to suit this.

* Network sizes changed.

* Test of environment variations added.

## Note

To run the code `train.py`, you need to copy the  [SVPG2023/GYMIP/env at main · yzlc080733/SVPG2023 · GitHub](https://github.com/yzlc080733/SVPG2023/tree/main/GYMIP/env) folder to replace the `env/` here.
