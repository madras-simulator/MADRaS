import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import rllib_helpers as helpers


helpers.register_madras()
ray.init()
config = ppo.DEFAULT_CONFIG.copy()
# Full config is here: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/agents/trainer.py#L42
config["num_gpus"] = 0
config["num_workers"] = 1
config["eager"] = False
config["vf_clip_param"] = 20  # originally it was 10. We should consider scaling down the rewards for keeping episode reward under 2000
# config["gamma"] = 0.7
# config["lr"] = 5e-7
# config["batch_mode"] = "complete_episodes"
# config["train_batch_size"] = 10000

trainer = ppo.PPOTrainer(config=config, env="madras_env")

# Can optionally call trainer.restore(path) to load a checkpoint.
# trainer.restore('/home/anirban/ray_results/PPO_madras_env_2019-12-03_08-56-11fe8guncf/checkpoint_1713/checkpoint-1713')
# trainer.restore('/home/anirban/ray_results/PPO_madras_env_2019-12-03_19-15-01h5u9w2bj/checkpoint_311/checkpoint-311')

# Alpine checkpoint from which Spring was restored:
# trainer.restore('/home/anirban/ray_results/PPO_madras_env_2019-12-04_06-36-12t0juqqsq/checkpoint_471/checkpoint-471')

# Good Spring checkpoint
# trainer.restore('/home/anirban/ray_results/PPO_madras_env_2019-12-04_09-37-1648nnevup/checkpoint_1802/checkpoint-1802')
# Trying to solve the issue of crash at 15813-15817 step range which never seems to appear during training
for i in range(10000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 10 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)