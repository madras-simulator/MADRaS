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


CHECKPOINT = "/home/anirban/ray_results/PPO_madras_env_2019-12-03_10-11-12ek55po4b/checkpoint_1734/checkpoint-1734"
os.system("rllib rollout {} --env Madras-v0 --steps 1000000 --run PPO --no-render".format(CHECKPOINT))
os.system("pkill torcs")