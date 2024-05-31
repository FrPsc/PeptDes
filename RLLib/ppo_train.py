import os
import numpy as np
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ENV.swap_env import PeptDes_SWAP

ray.init(num_gpus=2)

config = (
    PPOConfig()
    .training(gamma=0.99, lr=0.01, train_batch_size=3600)
    .rollouts(num_rollout_workers=36, rollout_fragment_length=100) #Use 36 parallel rollout workers
    .resources(num_gpus=2)
    .environment(env=PeptDes_SWAP)
)

algo = config.build()

for i in range(100): #100 training loops
    algo.train()
    algo.get_policy().export_checkpoint("CHECKPOINTS/") #Saves trained model
