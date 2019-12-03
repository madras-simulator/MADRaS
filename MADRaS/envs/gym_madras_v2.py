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
import utils.snakeoil3_gym_v2 as snakeoil3
from utils.gym_torcs_v2 import TorcsEnv
from controllers.pid import PIDController
import gym
from gym.utils import seeding
import os
import subprocess
import signal
import time
from mpi4py import MPI
import socket
import envs.config_parser as config_parser
import envs.reward_manager as rm
import envs.done_manager as dm
import envs.observation_manager as om
import traffic.traffic as traffic
import multiprocessing
import threading

DEFAULT_SIM_OPTIONS_FILE = "envs/data/sim_options_v2.yml"


class MadrasAgent(TorcsEnv, gym.Env):
    def __init__(self, name, torcs_server_port, cfg, sim_info={}, debug=False):
        self.debug = debug
        self._config = config_parser.MadrasAgentConfig()
        self.name = name
        for key in sim_info:
            cfg[key] = sim_info[key]
        self._config.update(cfg)
        self.torcs_server_port = torcs_server_port
        TorcsEnv.__init__(self,
                          name=self.name,
                          vision=self._config.vision,
                          throttle=self._config.throttle,
                          gear_change=self._config.gear_change)

        if self._config.pid_assist:
            self.PID_controller = PIDController(self._config.pid_settings)
            self.action_dim = 2  # LanePos, Velocity
        else:
            self.action_dim = 3  # Steer, Accel, Brake
        self.observation_manager = om.ObservationManager(self._config.observations,
                                                         self._config.vision)
        if self._config.normalize_actions:
            self.action_space = gym.spaces.Box(low=-np.ones(3), high=np.ones(3))

        self.observation_space = self.observation_manager.get_observation_space()
        self.obs_dim = self.observation_manager.get_state_dim()  # No. of sensors input
        self.state_dim = self.observation_manager.get_state_dim()
        self.reward_manager = rm.RewardManager(self._config.rewards)
        self.done_manager = dm.DoneManager(self._config.dones)
        self.initial_reset = True
        self.client = None

    def create_new_client(self):
        while True:
            try:
                self.client = snakeoil3.Client(p=self.torcs_server_port,
                                               name=self.name)
                # Open new UDP in vtorcs
                self.client.MAX_STEPS = self._config.client_max_steps
                if self.debug:
                    print("{} successfully created a client that connects to "
                        "server-PID {}".format(self.name, self.client.serverPID))
                break
            except Exception as e:
                print("{} received error {} during client creation.".format(self.name, e))
                pass

    def get_observation_from_server(self):
        self.client.get_servers_input(0)
        # Get the initial input from torcs
        raw_ob = self.client.S.d
        # Get the current full-observation from torcs
        self.ob = self.make_observation(raw_ob)
        if self.debug:
            print("[{}]: Initial observation: {}".format(self.name, self.ob))
        if np.any(np.asarray(self.ob.track) < 0):
            print("Reset produced bad track values.")
        self.distance_traversed = 0
        s_t = self.observation_manager.get_obs(self.ob, self._config)
        # return_dict[self.name] = s_t

        return s_t

    def complete_reset(self):
        if self._config.pid_assist:
            self.PID_controller.reset()
        self.reward_manager.reset()
        self.done_manager.reset()
        print("Reset: Starting new episode")


    def reset(self):
        """DEPRECATED"""
        if self.initial_reset:
            self.wait_for_observation()
            self.initial_reset = False
        else:
            while True:
                try:
                    self.ob, self.client = TorcsEnv.reset(self, client=self.client, relaunch=True)
                except Exception:
                    self.wait_for_observation()

                if not np.any(np.asarray(self.ob.track) < 0):
                    break
                else:
                    print("Reset: Reset failed as agent started off track. Retrying...")

        if np.any(np.asarray(self.ob.track) < 0):
            print("Reset produced bad track values.")
        self.distance_traversed = 0
        print("Initial observation: {}".format(self.ob))
        s_t = self.observation_manager.get_obs(self.ob, self._config)
        if self._config.pid_assist:
            self.PID_controller.reset()
        self.reward_manager.reset()
        self.done_manager.reset()
        print("Reset: Starting new episode")

        return s_t

    def wait_for_observation(self):
        """Refresh client and wait for a valid observation to come in."""
        self.ob = None
        while self.ob is None:
            if self.debug:
                print("{} Still waiting for observation".format(self.name))
            try:
                self.client = snakeoil3.Client(p=self.torcs_server_port,
                                               name=self.name)
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
            print("Exception {} caught at port {}".format(str(e), self.torcs_server_port))
            self.wait_for_observation()

        game_state = {"torcs_reward": r,
                      "torcs_done": done,
                      "distance_traversed": self.client.S.d['distRaced'],
                      "angle": self.client.S.d["angle"],
                      "damage": self.client.S.d["damage"],
                      "trackPos": self.client.S.d["trackPos"],
                      "track": self.client.S.d["track"]}
        reward = self.reward_manager.get_reward(self._config, game_state)

        done = self.done_manager.get_done_signal(self._config, game_state)

        next_obs = self.observation_manager.get_obs(self.ob, self._config)

        return next_obs, reward, done, info


    def step_pid(self, desire):
        """Execute single step with lane_pos, velocity controls."""
        if self._config.normalize_actions:
            # [-1, 1] should correspond to [-self._config.target_speed,
            #                                self._config.target_speed]
            speed_scale = self._config.target_speed
            desire[1] *= speed_scale  # Now in m/s
            # convert to km/hr
            desire[1] *= 3600/1000  # Now in km/hr
            # Normalize to gym_torcs specs
            desire[1] /= self.default_speed

        reward = 0.0

        for PID_step in range(self._config.pid_settings['pid_latency']):
            a_t = self.PID_controller.get_action(desire)
            try:
                self.ob, r, done, info = TorcsEnv.step(self, PID_step,
                                                       self.client, a_t,
                                                       self._config.early_stop)
            except Exception as e:
                print("Exception {} caught at port {}".format(str(e), self.torcs_server_port))
                self.wait_for_observation()
                #  TODO(santara): step not performed...

            game_state = {"torcs_reward": r,
                          "torcs_done": done,
                          "distance_traversed": self.client.S.d["distRaced"],
                          "angle": self.client.S.d["angle"],
                          "damage": self.client.S.d["damage"],
                          "trackPos": self.client.S.d["trackPos"],
                          "track": self.client.S.d["track"]}

            reward += self.reward_manager.get_reward(self._config, game_state)
            if self._config.pid_assist:
                self.PID_controller.update(self.ob)
            done = self.done_manager.get_done_signal(self._config, game_state)
            if done:
                break

        next_obs = self.observation_manager.get_obs(self.ob, self._config)
        return next_obs, reward, done, info

    def step(self, action, return_dict={}):
        if self._config.pid_assist:
            return_dict[self.name] = self.step_pid(action)
        else:
            return_dict[self.name] = self.step_vanilla(action)
        return return_dict


class MadrasEnv(gym.Env):
    """Definition of the Gym Madras Environment."""
    def __init__(self, cfg_path=DEFAULT_SIM_OPTIONS_FILE, debug=False):
        # If `visualise` is set to False torcs simulator will run in headless mode
        self.debug = debug
        self._config = config_parser.MadrasEnvConfig()
        self._config.update(config_parser.parse_yaml(cfg_path))
        self.torcs_proc = None
        self.seed()
        self.start_torcs_process()
        self.agents ={} 
        if self._config.agents:
            for i, agent in enumerate(self._config.agents):
                agent_name = [x for x in agent.keys()][0]
                agent_cfg = agent[agent_name]
                name = '{}_{}'.format(agent_name, i)
                torcs_server_port = self._config.torcs_server_port + i
                self.agents[name] = MadrasAgent(name, torcs_server_port, agent_cfg,
                                                {"track_len": self._config.track_len,
                                                 "max_steps": self._config.max_steps
                                                })
        if self._config.traffic:
            self.traffic_manager = traffic.MadrasTrafficManager(
                self._config.torcs_server_port, len(self.agents), self._config.traffic)
        # self.action_dim = self.agents[0].action_dim  # TODO(santara): Can I not have different action dims for different agents?
        self.initial_reset = True
        print("Madras agents are: ", self.agents)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def config(self):
        return self._config
        
    def test_torcs_server_port(self):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(('', self._config.torcs_server_port))
        except:
            print("Specified torcs_server_port {} is not available. "
                  "Searching for alternative...".format(
                  self._config.torcs_server_port))
            udp.bind(('', 0))
            _, self._config.torcs_server_port = udp.getsockname()
            print("torcs_server_port has been reassigned to {}".format(
                   self._config.torcs_server_port))

        udp.close()

    def start_torcs_process(self):
        if self.torcs_proc is not None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            time.sleep(0.5)
            self.torcs_proc = None
        self.test_torcs_server_port()
        self.execute_torcs_launch_command()
        time.sleep(1)

    def reset_torcs(self):
        print("Relaunching TORCS on port{}".format(self._config.torcs_server_port))
        # os.kill(self.torcs_proc.pid, signal.SIGKILL)
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
        time.sleep(1)
        self.execute_torcs_launch_command()
        time.sleep(1)

    def execute_torcs_launch_command(self):
        command = None
        rank = MPI.COMM_WORLD.Get_rank()

        if rank < self._config.no_of_visualisations and self._config.visualise:
            command = 'export TORCS_PORT={} && vglrun torcs -t 10000000 -nolaptime'.format(
                       self._config.torcs_server_port)
        else:
            command = 'export TORCS_PORT={} && torcs -t 10000000 -r ~/.torcs/config/raceman/quickrace.xml -nolaptime'.format(self._config.torcs_server_port)
        if self._config.vision is True:
            command += ' -vision'

        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        print("TORCS server PID is: ", self.torcs_proc.pid)

    def get_observation_from_agent(self, agent_num):
        agent_name = "MadrasAgent_{}".format(agent_num)
        return self.agents[agent_name].get_observation_from_server()

    def reset(self):
        """Reset Method to be called at the end of each episode."""
        if not self.initial_reset:
            self.reset_torcs()
        else:
            self.initial_reset = False
        if self._config.traffic:
            self.traffic_manager.reset()
        s_t = {}
        
        # Serial reset step by step : best working solution till date
        # Create clients and connect their sockets
        for agent in self.agents:
            self.agents[agent].create_new_client()
        # Collect first observations
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        agent_nums = [0, 1]
        result = p.map(self.get_observation_from_agent, (agent_nums,))
        p.close()
        p.join()
        print(result)
        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        # jobs = []
        # for agent in self.agents:
        #     # s_t[agent] = self.agents[agent].get_observation_from_server()
        #     p = multiprocessing.Process(target=self.agents[agent].get_observation_from_server, args=(return_dict))
        #     jobs.append(p)
        #     p.start()
        # for proc in jobs:
        #     proc.join()
        # for agent in self.agents:
        #     s_t[agent] = return_dict[agent]
        # Finish reset
        for agent in self.agents:
            self.agents[agent].complete_reset()

        return s_t

    def step(self, action):
        next_obs, reward, done, info = {}, {}, {}, {}
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i, agent in enumerate(self.agents):
            p = multiprocessing.Process(target=self.agents[agent].step, args=(action[i], return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        for agent in self.agents:
            next_obs[agent] = return_dict[agent][0]
            reward[agent] = return_dict[agent][1]
            done[agent] = return_dict[agent][2]
            info[agent] = return_dict[agent][3]

        return next_obs, reward, done, info
