"""
TODO(santara): write a python script to use rollout.py that does not depend on the Gym
registry of Madras-v0 

Steps for use:

1. Register Madras in OpenAI gym by adding the following snippet to
   /home/anirban/anaconda3/lib/python3.7/site-packages/gym/__init__.py or equivalent
```
import gym
gym.envs.register(
     id='Madras-v0',
     entry_point='envs:MadrasEnv',
)
```

2. Set checkpoint path
3. run: python eval_rllib_agent.py
"""


import os


# CHECKPOINT = "/home/anirban/ray_results/PPO_madras_env_2019-12-03_20-16-31w9ik1q41/checkpoint_1052/checkpoint-1052"
# Beautiful checkpoint for Spring:
# CHECKPOINT = "/home/anirban/ray_results/PPO_madras_env_2019-12-04_14-27-53f19t1utp/checkpoint_1893/checkpoint-1893"
CHECKPOINT = "/home/anirban/ray_results/PPO_madras_env_2019-12-04_20-03-37u4x3v1qw/checkpoint_61/checkpoint-61"
CHECKPOINT = "/home/anirban/ray_results/PPO_madras_env_2019-12-05_07-11-40l711dsji/checkpoint_31/checkpoint-31"
os.system("rllib rollout {} --env Madras-v0 --steps 1000000 --run PPO --no-render".format(CHECKPOINT))
os.system("pkill torcs")