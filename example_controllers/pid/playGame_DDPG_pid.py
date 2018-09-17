"""
DDPG Training.

PID Control = True
"""
import sys
sys.path.append("../../utils/")
sys.path.append("./DDPG/")
import os
import gc
import yaml
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from madras_datatypes import Madras
from gym_madras import MadrasEnv
from display_utils import *
from ddpg import *


np.random.seed(1337)
gc.enable()
figure = plt.figure()
madras = Madras()

with open("configurations.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

# config parameters are printed in main


def visualize_action_value(list_Q, fig, ep_no):
    """Heatmap Plotter."""
    actions_lane = []
    actions_vel = []
    Q_values = []
    for act, q in list_Q:
        actions_lane.append(act[0])
        actions_vel.append(act[1])
        Q_values.append(q[0])
    actions_vel = np.asarray(actions_vel)
    actions_lane = np.asarray(actions_lane)
    Q_values = np.asarray(Q_values)
    plot_heatmap(actions_lane, actions_vel, Q_values)
    # plt.subplot(211)
    # plt.clf()
    # plt.pause(1)
    # plt.scatter(actions_lane, Q_values, alpha=0.5)
    # plt.title("Lane_Pos")
    # plt.subplot(212)
    # plt.scatter(actions_vel, Q_values, alpha=0.5)
    # plt.title("Velocity")
    plt.savefig(os.path.join(cfg['configs']['fig_save_dir'],
                "plot_" + str(ep_no) + ".png"))
    # plt.show(block=False)
    # plt.pause(1)
    plt.clf()
    # plt.pause(1)
    # fig.clear()


def playGame(f_diagnostics, train_indicator, port=3101):
    """Training Method."""
    EXPLORE = cfg['agent']['total_explore']
    episode_count = cfg['agent']['max_eps']
    max_steps = cfg['agent']['max_steps_eps']
    EPSILON = cfg['agent']['epsilon_start']
    DELTA = cfg['agent']['delta']
    done = False
    epsilon_steady_state = 0.01  # This is used for early stopping.
    running_average_distance = 0.
    max_distance = 0.
    best_reward = cfg['agent']['start_reward']
    running_avg_reward = 0.
    weights_location = cfg['configs']['save_location'] + str(port) + "/"
    episode_printer = StructuredPrinter(mode="episode")
    step_printer = StructuredPrinter(mode="step")
    # Set up a Torcs environment
    print(("TORCS has been asked to use port: ", port))
    env = MadrasEnv(vision=False, throttle=True, gear_change=False,
                    port=port, pid_assist=True)

    agent = DDPG(env.env_name, env.state_dim, env.action_dim, weights_location,
                 cfg['agent']['reward_threshold'], cfg['agent']['thresh_coin_toss'])

    s_t = env.reset()
    print("\n\n-----------------Experiment Begins-----------------------\n\n")

    for i in range(episode_count):

        save_indicator = 1
        temp_distance = 0.
        total_reward = 0.
        info = {'termination_cause': 0}
        distance_traversed = 0.
        speed_array = []
        trackPos_array = []
        dmg = 0.0
        dmg_S = 0.0
        discount = 0.1
        CRASH_COUNTER = 0
        SPEED_BUFFER_SIZE = cfg['agent']['speed_buffer_size']
        count_stop = 0
        temp_replay_buffer = []

        for step in range(max_steps):
            dst = env.distance_traversed

            if (train_indicator):

                    step_printer.data["Dist_Raced"] = dst
                    if dst < running_average_distance:
                        epsilon = EPSILON - DELTA
                        step_printer.data["eps"] = epsilon
                        step_printer.explore_mode = -1
                    elif dst > max_distance:
                        epsilon = EPSILON + DELTA
                        step_printer.data["eps"] = epsilon
                        step_printer.explore_mode = 1
                    else:
                        epsilon = deepcopy(EPSILON)
                        step_printer.data["eps"] = epsilon
                        step_printer.explore_mode = 0
                    EPSILON -= 1.0 / EXPLORE
                    EPSILON = max(EPSILON, epsilon_steady_state)
                    d_t, step_info = agent.noise_action(s_t, epsilon)
                    for k in list(step_info.data.keys()):
                        step_printer.data[k] = step_info.data[k]
                    # Giving the car an inital kick
                    if step <= 50:
                        d_t[1] = abs(d_t[1]) + 0.1

            else:  # Testing phase
                d_t = agent.action(s_t)
            try:
                s_t1, r_t, done, info = env.step(d_t)
                distance_traversed += env.distance_traversed
                speed_array.append(env.ob.speedX * np.cos(env.ob.angle))
                trackPos_array.append(env.ob.trackPos)
            # Add to replay buffer only if training
                if train_indicator:
                    temp_replay_buffer.append([s_t, d_t, r_t, s_t1, done])
                total_reward += r_t
                s_t = s_t1
                dmg_S = 0.1 * dmg_S + (env.ob.damage - dmg) * 10
                if not dmg == env.ob.damage:
                    CRASH_COUNTER = CRASH_COUNTER + 1
                dmg = env.ob.damage
                step_printer.data["Desired_Trackpos"] = d_t[0]
                step_printer.data["Desired_Velocity"] = d_t[1]
                step_printer.data["Reward"] = r_t
                step_printer.print_step(step)

            except:
                pass

            if done:
                temp_distance = dst
                if temp_distance > max_distance:
                    max_distance = temp_distance
                break

        if train_indicator:
            episode_info = agent.perceive(temp_replay_buffer, total_reward)
            # Train agent

        for k in list(episode_info.data.keys()):
            episode_printer.data[k] = episode_info.data[k]

        temp_replay_buffer = []

        # Saving the best model yet
        if save_indicator == 1 and train_indicator == 1:
            if best_reward < total_reward:
                if(best_reward < total_reward):
                    best_reward = total_reward
                agent.saveNetwork(port)

        running_avg_reward = running_average(running_avg_reward,
                                             i + 1, total_reward)
        running_average_distance = running_average(running_average_distance,
                                                   i + 1, temp_distance)
        temp_distance = 0.

        episode_printer.data["Total_Steps"] = step
        episode_printer.data["Agv_Speed"] = np.mean(speed_array)
        episode_printer.data["Avg_TrackPos"] = np.mean(trackPos_array)
        episode_printer.data["Dist_Traversed"] = distance_traversed
        episode_printer.data["Traj_Reward"] = total_reward
        episode_printer.data["Run_Avg_Traj_Reward"] = running_avg_reward
        episode_printer.data["Run_Avg_Dist_Trav"] = running_average_distance
        episode_printer.data["Max_Dist_Trav"] = max_distance
        episode_printer.data["Replay_Buffer_Size"] = agent.replay_buffer.num_experiences

        episode_printer.print_episode(i)
        '''
        agent.visualize = []
        for data in agent.replay_buffer.getBatch(500):#use getBatch(500)
        here to reduce latency
            action = np.asarray(data.action)
            state = np.asarray(data.state)
            agent.visualize.append((action,agent.critic_network.q_value(state.reshape((1,agent.state_dim)),action.reshape((1,agent.action_dim)))))
        visualize_action_value(agent.visualize,figure,i)
        agent.visualize = []
        '''
        s_t = env.reset(info)
        """ document_episode(i, distance_traversed, speed_array, trackPos_array,
        info, running_avg_reward, f_diagnostics)
        """

    env.end()  # This is for shutting down TORCS-engine
    plt.close(figure)
    print("Finished.")


def document_episode(episode_no, distance_traversed, speed_array,
                     trackPos_array, info, running_avg_reward, f_diagnostics):
    """
    Note down a tuple of diagnostic values for each episode.

    (episode_no, distance_traversed, mean(speed_array), std(speed_array),
     mean(trackPos_array), std(trackPos_array), info[termination_cause],
     running_avg_reward)
    """
    f_diagnostics.write(str(episode_no) + ",")
    f_diagnostics.write(str(distance_traversed) + ",")
    f_diagnostics.write(str(np.mean(speed_array)) + ",")
    f_diagnostics.write(str(np.std(speed_array)) + ",")
    f_diagnostics.write(str(np.mean(trackPos_array)) + ",")
    f_diagnostics.write(str(np.std(trackPos_array)) + ",")
    f_diagnostics.write(str(running_avg_reward) + "\n")


def running_average(prev_avg, num_episodes, new_val):
    """Running average compute."""
    total = prev_avg * (num_episodes - 1)
    total += new_val
    return madras.floatX(total / num_episodes)

def analyse_info(info, printing=True):
    """Print Helper Function"""
    simulation_state = ['Normal', 'Terminated as car is OUT OF TRACK',
                        'Terminated as car has SMALL PROGRESS',
                        'Terminated as car has TURNED BACKWARDS']
    if printing and info['termination_cause'] != 0:
        print((simulation_state[info['termination_cause']]))

if __name__ == "__main__":

    try:
        port = int(sys.argv[1])
    except Exception as e:
        print("Usage : python %s <port>" % (sys.argv[0]))
        sys.exit()

    print('is_training : ' + str(cfg['configs']['is_training']))
    print('Starting best_reward : ' + str(cfg['agent']['start_reward']))
    print((cfg['agent']['total_explore']))
    print((cfg['agent']['max_eps']))
    print((cfg['agent']['max_steps_eps']))
    print((cfg['agent']['epsilon_start']))
    print('config_file : ' + cfg['configs']['configFile'])
    f_diagnostics = open("logger_17June.log", 'w')
    f_diagnostics.write("EPISODE,distance_traversed,mean_speed,std_speed,\
                         mean_track,std_track,reward_runn\n")
    playGame(f_diagnostics, train_indicator=cfg['configs']['is_training'],
             port=port)
    f_diagnostics.close()
