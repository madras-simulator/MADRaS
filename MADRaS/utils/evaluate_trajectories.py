"""
Utilities for analysing trajectories rolled out by MADRaS/agents/rllib/eval_rllib_agent.py

IMPORTANT:
Edit <path to python3 installation directory>/site-packages/ray/rllib/rollout.py as 
https://gist.github.com/Santara/955788aefac41645b135286d4a9b8633 to include `info` in the
saved trajectories.

"""

import numpy as np
import yaml
import pickle as pkl
import MADRaS.utils.data.track_details as track_details

DATA_PATH = "<path to evaluation directory>/eval_trajs.pkl"
DEFAULT_SPEED = 50  # kmph. replicate value in gym_torcs

def parse_yml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

class TrajEvaluator(object):
    def __init__(self, track_name='alpine-1', data_path=None):
        if not data_path:
            data_path = DATA_PATH
        with open(data_path, "rb") as f:
            self.traj_data = pkl.load(f)
        self.track_name = track_name
        track_lengths = track_details.track_lengths
        if not self.track_name or self.track_name not in track_lengths:
            raise ValueError("Unknown track name {}".format(self.track_name))
        self.track_length = track_lengths[self.track_name]
        print("{} trajectories loaded for evaluation".format(len(self.traj_data)))

    @property
    def num_trajs(self):
        return len(self.traj_data)

    @property
    def avg_dist_raced_before_done(self):
        dist_raced = []
        for traj in self.traj_data:
            dist_raced.append(traj[-1][5]["distRaced"])
        return np.mean(dist_raced)

    @property
    def avg_frac_track_covered(self):
        return self.avg_dist_raced_before_done/self.track_length

    @property
    def avg_speed(self):
        speeds = []
        for traj in self.traj_data:
            for transition in traj:
                speedX = transition[0][-3]
                speedY = transition[0][-2]
                speed = np.sqrt(speedX**2 + speedY**2)
                speeds.append(speed)
        return np.mean(speeds)*DEFAULT_SPEED

    @property
    def avg_reward(self):
        rewards = []
        for traj in self.traj_data:
            traj_reward = []
            for transition in traj:
                reward = transition[3]
                traj_reward.append(reward)
            rewards.append(np.sum(traj_reward))
        return np.mean(rewards)

    @property
    def race_complete(self):
        dist_raced = []
        for traj in self.traj_data:
            dist_raced.append(traj[-1][5]["distRaced"])
        dist_raced = np.asarray(dist_raced)
        return np.sum(dist_raced >= self.track_length)/len(dist_raced)

    @property
    def rank_one_fraction(self):
        num_rank_one_trajs = 0
        for traj in self.traj_data:
            racePos = traj[-1][5]["racePos"]
            if racePos == 1:
                num_rank_one_trajs += 1
        return float(num_rank_one_trajs/self.num_trajs)
        

def evaluate_overtaking_policy():
    data_path = ""
    track = "aalborg"
    evaluator = TrajEvaluator(track_name=track, data_path=data_path)
    print("The agent successfully overtook all cars {}% times".format(100*evaluator.rank_one_fraction))


def evaluate_in_batch():
    OUTFILE = open("<path to evaluation directory>/results.txt", 'a')
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
        "alpine-1",
        "spring"
    ]

    RESULTS_TEMPLATE = """
    ==================================================
        Car: {}\n
        Track: {}\n
        Average distance covered: {}%\n
        Average speed: {}\n
        Successful race completion rate: {}\n
    ==================================================
    """

    for track in TRACKS:
        for car in CARS:
            print("\n\nCAR: {}\n\n".format(car))
            data_path = "<path to evaluation directory>/eval_trajs_{}_{}.pkl".format(car, track)
            evaluator = TrajEvaluator(track_name=track, data_path=data_path)
            OUTFILE.write(RESULTS_TEMPLATE.format(car, track, evaluator.avg_frac_track_covered, evaluator.avg_speed, evaluator.race_complete))
            print(evaluator.avg_dist_raced_before_done)
    OUTFILE.close()




if __name__=='__main__':
    evaluate_in_batch()

    # OTHER WAYS OF USING THE UTILITIES:

    # evaluator = TrajEvaluator(track_name='alpine-1')
    # print("Average trajectory reward: {}".format(evaluator.avg_reward))
    # print("Average speed: {} kmph".format(evaluator.avg_speed))
    # print("Average fraction of track covered before done: {} %".format(100*evaluator.avg_frac_track_covered))
    # print("Successful race completion rate: {}".format(evaluator.race_complete))

    # evaluate_overtaking_policy()