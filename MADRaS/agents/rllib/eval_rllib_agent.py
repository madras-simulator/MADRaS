"""
Steps for use:

1. Set checkpoint path
2. run: python eval_rllib_agent.py
"""


import os
from MADRaS.utils.evaluate_trajectories import TrajEvaluator

SIM_OPTIONS_TEMPLATE = """
# TODO(santara) Each car model should have its own PID parameters, assign track length and track width correctly for each track
torcs_server_port: 60934
server_config:
  max_cars: 1
  track_names:
    - {}
  distance_to_start: 25
  torcs_server_config_dir: /home/anirban/.torcs/config/raceman/  # This must be the value for TORCS with rendering on
  scr_server_config_dir: /home/anirban/usr/local/share/games/torcs/drivers/scr_server/
  traffic_car: p406           # get full list of cars here: /home/anirban/usr/local/share/games/torcs/cars/
  learning_car:               # get full list of cars here: /home/anirban/usr/local/share/games/torcs/cars/
    - {}


randomize_env: False
add_noise_to_actions: False
action_noise_std: 0.1  # Only in effect when add_noise_to_actions is True
noisy_observations: False  # Adds sensor noise. See Section 7.7 of the scr_server paper: https://arxiv.org/abs/1304.1672

vision: False  # whether to use vision input
throttle: True
gear_change: False
client_max_steps: -1  # to be interpreted as np.inf
visualise: False  # whether to render TORCS window
no_of_visualisations: 1  # To visualize multiple training instances (under MPI) in parallel set it to more than 1
track_len: 6355.65  # in metres. All track lengths can be found here: http://www.berniw.org/trb/tracks/tracklist.php
max_steps: 20000 #15000 #20000  # max episode length
track_width: 12.0
target_speed: 27.78 # 13.89  # metres per sec
state_dim: 60
early_stop: True
normalize_actions: True  # all actions in [-1, 1]

# PID params
pid_assist: True
pid_settings:
  accel_pid:
    - 10.5  # a_x
    - 0.05  # a_y
    - 2.8   # a_z
  steer_pid:
    - 5.1
    - 0.001
    - 0.000001
  accel_scale: 1.0
  steer_scale: 0.1
  pid_latency: 5

# Observation mode
observations:
  mode: SingleAgentSimpleLapObs
  normalize: False  # gym_torcs normalizes by default
  obs_min:
    angle: -3.142
    track: 0.0
    trackPos: -1.0
  obs_max:
    angle: 3.142
    track: 200.0
    trackPos: 1.0

# Reward function
rewards:
  ProgressReward2:
    scale: 1.0
  AvgSpeedReward:
    scale: 1.0
  CollisionPenalty:
    scale: 10.0
  TurnBackwardPenalty:
    scale: 10.0
  AngAcclPenalty:
    scale: 5.0
    max_ang_accl: 2.0

# Done function
dones:
  - RaceOver
  - TimeOut
  - Collision
  - TurnBackward
  - OutOfTrack

"""



CARS = [
     "car1-stock1",
     "car1-stock2",
     "155-DTM",
     "car3-trb1",
     "kc-2000gt",
     "buggy",
     "baja-bug"
]

TRACKS = [
    "aalborg",
    "alpine-1",
    "alpine-2",
    "brondehach",
    "g-track-1",
    "g-track-2",
    "g-track-3",
    "corkscrew",
    "eroad",
    "e-track-2",
    "e-track-3",
    "e-track-4",
    "e-track-6",
    "forza",
    "ole-road-1",
    "ruudskogen",
    "street-1",
    "wheel-1",
    "wheel-2",
    "spring"
]

CHECKPOINT = "<path to checkpoint>"


def log_results_single_agent_no_traffic():

    RESULTS_TEMPLATE = """
    ==================================================
        Car: {}\n
        Track: {}\n
        Average distance covered: {}\n
        Average speed: {}\n
        Successful race completion rate: {}\n
        Num trajectories evaluated: {}\n
    ==================================================
    """
    OUTFILE = open("<path to evaluation directory>/results.txt", 'a')
    OUTFILE.write("Evaluating {} on {} in {}\n\n".format(CHECKPOINT, CARS, TRACKS))
    SIM_OPTIONS_PATH = "<path to MADRaS root directory>/MADRaS/envs/data/madras_config.yml"

    for track in TRACKS:
        for car in CARS:
              print("\n\nCAR: {}\n\n".format(car))
              with open(SIM_OPTIONS_PATH, "w") as f:
                  f.write(SIM_OPTIONS_TEMPLATE.format(track, car))
              data_path = "<path to evaluation directory>/eval_trajs_{}_{}.pkl".format(car, track)
              os.system("rllib rollout {} --env Madras-v0 --steps 1000000 --run PPO --no-render --out {}".format(CHECKPOINT, data_path))
              os.system("pkill torcs")
              evaluator = TrajEvaluator(track_name=track, data_path=data_path)
              OUTFILE.write(RESULTS_TEMPLATE.format(car, track, evaluator.avg_frac_track_covered, evaluator.avg_speed,
                                                    evaluator.race_complete, evaluator.num_trajs))
              
    OUTFILE.close()


def log_results_single_agent_in_traffic():

    RESULTS_TEMPLATE = """
    ==================================================
        Successful race completion rate: {}\n
        Num trajectories evaluated: {}\n
    ==================================================
    """

    OUTFILE = open("<path to evaluation directory>/results.txt", 'a')
    OUTFILE.write("Evaluating {} on 2 traffic agents\n\n".format(CHECKPOINT))
    data_path = "<path to evaluation directory>/eval_trajs.pkl"
    os.system("rllib rollout {} --env Madras-v0 --steps 7000 --run PPO --no-render --out {}".format(CHECKPOINT, data_path))
    os.system("pkill torcs")
    os.system("pkill torcs")
    evaluator = TrajEvaluator(data_path=data_path)
    OUTFILE.write(RESULTS_TEMPLATE.format(evaluator.rank_one_fraction, evaluator.num_trajs))
    print(RESULTS_TEMPLATE.format(evaluator.rank_one_fraction, evaluator.num_trajs))
              
    OUTFILE.close()


if __name__=='__main__':
    log_results_single_agent_in_traffic()
#     log_results_single_agent_no_traffic()