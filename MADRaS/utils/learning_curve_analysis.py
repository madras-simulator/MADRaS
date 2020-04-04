"""
A set of utilities for plotting learning curves.

DISCLAIMER: The field names are currently those of rllib. For use with other learning frameworks,
change the names accordingly.

To plot mean episode reward over training iterations, assign the right value to TRAINING_DIR and
run this script directly.

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

TRAINING_DIR = "/home/<username>/ray_results/<experiment name>"

class SingleLearningCurveAnalyser(object):
    def __init__(self, training_log_path=os.path.join(TRAINING_DIR, "progress.csv")):
        self.training_log_path = training_log_path
        self.training_log_data = pd.read_csv(self.training_log_path)

    def plot_over_training_iter(self, field="episode_reward_mean", training_iteration_range=None):
        if training_iteration_range is None:
            x = self.training_log_data["training_iteration"]
            y = self.training_log_data[field]
            plt.plot(x, y)
            plt.grid()
            plt.xlabel("training_iteration", fontsize=15)
            plt.ylabel(field, fontsize=15)
        else:
            begin_iter, end_iter = training_iteration_range
            df = self.training_log_data.loc[(self.training_log_data["training_iteration"] >= begin_iter)
                                            & (self.training_log_data["training_iteration"] <= end_iter)]
            x = df["training_iteration"]
            y = df[field]
            plt.plot(x, y)

def action_and_observation_noise_corkscrew():
    """
    Example of how the SingleLearningCurveAnalyser class can be used to generate comparison plots.

    The csv files corressponding to each of the training sessions to be compared must  be kept in
    one location `root_dir`.
    """
    root_dir = "<fill this up>"
    _, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    no_noise = SingleLearningCurveAnalyser(os.path.join(root_dir, "No_Noise2.csv"))
    obs_noise = SingleLearningCurveAnalyser(os.path.join(root_dir, "Only_obs_noise.csv"))
    action_noise_0_1 = SingleLearningCurveAnalyser(os.path.join(root_dir, "Action_noise_0.1.csv"))
    action_noise_0_5 = SingleLearningCurveAnalyser(os.path.join(root_dir, "Action_noise_0.5.csv"))
    obs_and_action_noise_0_1 = SingleLearningCurveAnalyser(os.path.join(root_dir, "Observation_and_Action_noise_0.1.csv"))

    no_noise.plot_over_training_iter(training_iteration_range=(1, 1000))
    obs_noise.plot_over_training_iter(training_iteration_range=(1, 1000))
    action_noise_0_1.plot_over_training_iter(training_iteration_range=(1, 1000))
    action_noise_0_5.plot_over_training_iter(training_iteration_range=(1, 1000))
    obs_and_action_noise_0_1.plot_over_training_iter(training_iteration_range=(1, 1000))
    plt.legend([
        "No noise",
        "Observation Noise",
        "Action Noise std=0.1",
        "Action Noise std=0.5",
        "Observation and Action Noise",
    ], fontsize=15)
    plt.xlabel("training iteration", fontsize=15)
    plt.ylabel("trajectory reward", fontsize=15)
    plt.grid()
    plt.show()

def curriculum_learning_analyser_spring():
    """
    Example of how the SingleLearningCurveAnalyser class can be used to generate comparison plots.

    The csv files corressponding to each of the training sessions to be compared must  be kept in
    one location `root_dir`.
    """
    root_dir = "<fill this up>"
    _, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    training_from_scratch_on_spring = SingleLearningCurveAnalyser(os.path.join(root_dir, "training_from_scratch.csv"))
    training_from_scratch_on_spring.plot_over_training_iter(training_iteration_range=(1, 2500))
    pretraining_alpine = SingleLearningCurveAnalyser(os.path.join(root_dir, "pre-training-in-alpine-1.csv"))
    pretraining_alpine.plot_over_training_iter(training_iteration_range=(0, 701))
    finetuning_alpine_on_spring = SingleLearningCurveAnalyser(os.path.join(root_dir, "finetuning_from_alpine-1.csv"))
    finetuning_alpine_on_spring.plot_over_training_iter(training_iteration_range=(702, 2500))
    pretraining_corkscrew = SingleLearningCurveAnalyser(os.path.join(root_dir, "pre-training-in-corkscrew.csv"))
    pretraining_corkscrew.plot_over_training_iter(training_iteration_range=(1, 561))
    finetuning_corkscrew_on_spring = SingleLearningCurveAnalyser(os.path.join(root_dir, "finetuning_from_corkscrew.csv"))
    finetuning_corkscrew_on_spring.plot_over_training_iter(training_iteration_range=(562, 2500))
    plt.legend(["scratch",
    "alpine-pretrain",
    "alpine-finetuning",
    "corkscrew-pretrain",
    "corkscrew-finetuning"], fontsize=15)
    plt.grid()
    plt.xlabel("training iterations", fontsize=15)
    plt.ylabel("episode reward mean", fontsize=15)
    plt.show()

if __name__=="__main__":
    analyser = SingleLearningCurveAnalyser()
    analyser.plot_over_training_iter()
    plt.show()