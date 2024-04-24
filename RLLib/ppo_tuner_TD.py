import os
import numpy as np
import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ENV.swap_env import PeptDes_SWAP

ray.init(num_gpus=2)

config = (
    PPOConfig()
    .training(gamma=0.99, clip_param=0.2, lr=tune.grid_search(np.arange(0.0001, 0.01, 0.001)), train_batch_size=256, sgd_minibatch_size=16, use_critic=False, use_gae=False, use_kl_loss=False)
    #.training(gamma=0.99, lr=tune.grid_search(np.arange(0.000001, 0.00031, 0.000031)), train_batch_size=256, sgd_minibatch_size=32)
    .rollouts(num_rollout_workers=1, rollout_fragment_length=1)
    .environment(env=PeptDes_SWAP)
)
config['batch_mode'] = 'complete_episodes'

#config['model']['fcnet_activation'] = 'relu'
#config['model']['fcnet_hiddens'] = [256,128]
print(config['model'])

tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={"training_iteration": 50},
    ),
    param_space=config,
)

results = tuner.fit()

best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
print(best_result)
