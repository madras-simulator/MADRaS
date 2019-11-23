"""
Gym Madras Env Wrapper.

This is an OpenAI gym environment wrapper for the MADRaS simulator. For more information on the OpenAI Gym interface, please refer to: https://gym.openai.com

Built on top of gym_torcs https://github.com/ugo-nama-kun/gym_torcs/blob/master/gym_torcs.py

The following enhancements were made for Multi-agent synchronization using exception handling:
- All the agents connect to the same TORCS engine through UDP ports
- If an agent fails to connect to the TORCS engine, it keeps on trying in a loop until successful
- Restart the episode for all the agents when any one of the learning agents terminates its episode

"""

import math
from copy import deepcopy
import numpy as np
import utils.snakeoil3_gym as snakeoil3
from utils.gym_torcs import TorcsEnv
from controllers.pid import PIDController
import gym
from gym.utils import seeding
import os
import subprocess
import signal
import time
from mpi4py import MPI
import socket
import yaml
import envs.reward_manager as rm
import envs.done_manager as dm
import envs.observation_manager as om

DEFAULT_SIM_OPTIONS_FILE = "envs/data/sim_options.yml"

class MadrasConfig(object):
    """Configuration class for MADRaS Gym environment."""
    def __init__(self):
        self.vision = False
        self.throttle = True
        self.gear_change = False
        self.port = 6006
        self.pid_assist = False
        self.pid_settings = {}
        self.client_max_steps = np.inf
        self.visualise = False
        self.no_of_visualisations = 1
        self.track_len = 7014.6
        self.max_steps = 20000
        self.target_speed = 15.0
        self.state_dim = 29
        self.normalize_actions = False
        self.early_stop = True
        self.observations = None
        self.rewards = {}
        self.dones = {}

    def update(self, cfg_dict):
        """Update the configuration terms from a dictionary.
        
        Args:
            cfg_dict: dictionary whose keys are the names of class attributes whose
                      values must be updated
        """
        if cfg_dict is None:
            return
        direct_attributes = ['vision', 'throttle', 'gear_change', 'port', 'pid_assist',
                             'pid_latency', 'visualise', 'no_of_visualizations', 'track_len',
                             'max_steps', 'target_speed', 'state_dim', 'early_stop', 'accel_pid',
                             'steer_pid', 'normalize_actions', 'observations', 'rewards', 'dones',
                             'pid_settings']
        for key in direct_attributes:
            if key in cfg_dict:
                exec("self.{} = {}".format(key, cfg_dict[key]))
        self.client_max_steps = (np.inf if cfg_dict['client_max_steps'] == -1
                                 else cfg_dict['client_max_steps'])
        self.validate()

    def validate(self):
        assert self.vision == False, "Vision input is not yet supported."
        assert self.throttle == True, "Throttle must be True."
        assert self.gear_change == False, "Only automatic transmission is currently supported."
        # TODO(santara): add checks for self.state_dim
        

def parse_yaml(yaml_file):
    if not yaml_file:
        yaml_file = DEFAULT_SIM_OPTIONS_FILE
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


class MadrasEnv(TorcsEnv, gym.Env):
    """Definition of the Gym Madras Environment."""
    def __init__(self, cfg_path=None):
        # If `visualise` is set to False torcs simulator will run in headless mode
        """Init Method."""
        self._config = MadrasConfig()
        self._config.update(parse_yaml(cfg_path))
        self.torcs_proc = None
        if self._config.pid_assist:
            self.action_dim = 2  # LanePos, Velocity
        else:
            self.action_dim = 3  # Steer, Accel, Brake
        self.observation_manager = om.ObservationManager(self._config.observations)
        self.reward_manager = rm.RewardManager(self._config.rewards)
        self.done_manager = dm.DoneManager(self._config.dones)
        TorcsEnv.__init__(self,
                          vision=self._config.vision,
                          throttle=self._config.throttle,
                          gear_change=self._config.gear_change,
                          visualise=self._config.visualise,
                          no_of_visualisations=self._config.no_of_visualisations)
        self.state_dim = self._config.state_dim  # No. of sensors input
        self.env_name = 'Madras_Env'
        self.client_type = 0  # Snakeoil client type
        self.initial_reset = True
        if self._config.pid_assist:
            self.PID_controller = PIDController(self._config.pid_settings)
        self.ob = None
        self.seed()
        self.start_torcs_process()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def config(self):
        return self._config
        
    def get_free_udp_port(self):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(('', 0))
        addr, port = udp.getsockname()
        udp.close()
        return port

    def start_torcs_process(self):
        if self.torcs_proc is not None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            time.sleep(0.5)
            self.torcs_proc = None

        self._config.port = self.get_free_udp_port()
        window_title = str(self._config.port)
        command = None
        rank = MPI.COMM_WORLD.Get_rank()

        if rank < self._config.no_of_visualisations and self._config.visualise:
            command = 'export TORCS_PORT={} && vglrun torcs -t 10000000 -nolaptime'.format(self._config.port)
        else:
            command = 'export TORCS_PORT={} && torcs -t 10000000 -r ~/.torcs/config/raceman/quickrace.xml -nolaptime'.format(self._config.port)
        if self._config.vision is True:
            command += ' -vision'

        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        time.sleep(1)

   
    def reset(self, prev_step_info=None):
        """Reset Method to be called at the end of each episode."""
        if self.initial_reset:
            self.wait_for_observation()
            self.initial_reset = False

        else:
            try:
                self.ob, self.client = TorcsEnv.reset(self, client=self.client, relaunch=True)
            except Exception:
                self.wait_for_observation()

        self.distance_traversed = 0
        s_t = self.observation_manager.get_obs(self.ob, self._config)
        if self._config.pid_assist:
            self.PID_controller.reset()
        self.reward_manager.reset()
        self.done_manager.reset()
        return s_t

    def wait_for_observation(self):
        """Refresh client and wait for a valid observation to come in."""
        self.ob = None
        while self.ob is None:
            try:
                self.client = snakeoil3.Client(p=self._config.port,
                                               vision=self._config.vision,
                                               visualise=self._config.visualise)
                # Open new UDP in vtorcs
                self.client.MAX_STEPS = self._config.client_max_steps
                self.client.get_servers_input(0)
                # Get the initial input from torcs
                raw_ob = self.client.S.d
                # Get the current full-observation from torcs
                self.ob = self.make_observation(raw_ob)
            except:
                pass

    def step_vanilla(self, action):
        """Execute single step with steer, acceleration, brake controls."""
        if self._config.normalize_actions:
            action[1] = (action[1] + 1) / 2.0  # acceleration back to [0, 1]
            action[2] = (action[2] + 1) / 2.0  # brake back to [0, 1]

        r = 0.0
        try:
            self.ob, r, done, info = TorcsEnv.step(self, 0,
                                                   self.client, action,
                                                   self._config.early_stop)
        except Exception as e:
            print(("Exception {} caught at port {}".format(str(e), self._config.port)))
            self.wait_for_observation()
        game_state = {"torcs_reward": r,
                      "torcs_done": done,
                      "distance_traversed": self.client.S.d['distRaced'],
                      "angle": self.client.S.d["angle"],
                      "damage": self.client.S.d["damage"]}
        reward = self.reward_manager.get_reward(self._config, game_state)

        done = self.done_manager.get_done_signal(self._config, game_state)

        next_obs = self.observation_manager.get_obs(self.ob, self._config)

        if done:
            self.client.R.d["meta"] = True  # Terminate the episode
            print('Terminating PID {}'.format(self.client.serverPID))

        return next_obs, reward, done, info


    def step_pid(self, desire):
        """Execute single step with lane_pos, velocity controls."""
        if self._config.normalize_actions:
            speed_scale = 2.0 * self._config.target_speed
            desire[1] *= speed_scale

        reward = 0.0

        for PID_step in range(self._config.pid_settings['pid_latency']):
            a_t = self.PID_controller.get_action(desire)
            try:
                self.ob, r, done, info = TorcsEnv.step(self, PID_step,
                                                       self.client, a_t,
                                                       self._config.early_stop)
            except Exception as e:
                print(("Exception {} caught at port {}".format(str(e), self._config.port)))
                self.wait_for_observation()
            game_state = {"torcs_reward": r,
                          "torcs_done": done,
                          "distance_traversed": self.client.S.d["distRaced"],
                          "angle": self.client.S.d["angle"],
                          "damage": self.client.S.d["damage"]}
            reward += self.reward_manager.get_reward(self._config, game_state)
            if self._config.pid_assist:
                self.PID_controller.update(self.ob)
            done = self.done_manager.get_done_signal(self._config, game_state)
            if done:
                self.client.R.d["meta"] = True  # Terminate the episode
                print('Terminating PID {}'.format(self.client.serverPID))
                break

        next_obs = self.observation_manager.get_obs(self.ob, self._config)

        return next_obs, reward, done, info

    def step(self, action):
        if self._config.pid_assist:
            return self.step_pid(action)
        else:
            return self.step_vanilla(action)