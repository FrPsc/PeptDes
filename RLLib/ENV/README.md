Python implementations for RL rely on Gymnasium as a standard API for RL environments: https://gymnasium.farama.org/index.html

A Custom environment requires:
- Definition of action space and observation space as attributes in __init__
- step method: receives action, applies it to the environment state and calculates reward
- reset method: initiate an episode, so it is called at the very beginning and every time step returns a termination signal
