from collections import OrderedDict
import numpy as np
from MADRaS.utils.gym_torcs import TorcsEnv
import MADRaS.utils.snakeoil3_gym as snakeoil3
from MADRaS.controllers.pid import PIDController
import MADRaS.utils.madras_datatypes as md
from multiprocessing import Process
from time import time
import logging
logger = logging.getLogger(__name__)

MadrasDatatypes = md.MadrasDatatypes()

class MadrasTrafficHandler(object):
    """Creates the traffic agents for a given training configuration."""
    def __init__(self, torcs_server_port, num_learning_agents, cfg):
        self.traffic_agents = OrderedDict()
        self.traffic_processes = []
        self.num_episodes_of_training = 0
        for i, agent in enumerate(cfg):
            agent_type = [x for x in agent.keys()][0]  # TODO(santara): find a better way of extracting key from a dictionary with a single entry
            agent_config = agent[agent_type]
            agent_name = '{}_{}'.format(agent_type, i)
            agent_port = torcs_server_port + i #+ num_learning_agents
            try:
                exec("self.traffic_agents['{}'] = {}({}, {}, '{}')".format(
                    agent_name, agent_type, agent_port, agent_config, agent_name))
            except Exception as e:
                raise ValueError("Unknown traffic class {} or incomplete specification.\n Original error: {}".format(
                                 agent_type, e))
    
    def get_traffic_agents(self):
        return self.traffic_agents

    def flag_off_traffic(self, num_cars_to_reset):
        self.num_episodes_of_training += 1
        for i, agent in enumerate(self.traffic_agents.values()):
            if i < num_cars_to_reset:
                self.traffic_processes.append(Process(target=agent.flag_off, args=(i+self.num_episodes_of_training,)))
        for traffic_process in self.traffic_processes:
            traffic_process.start()

    def kill_all_traffic_agents(self):
        for traffic_process in self.traffic_processes:
            traffic_process.terminate()
        self.traffic_processes = []

    def reset(self, num_cars_to_reset):
        self.kill_all_traffic_agents()
        self.flag_off_traffic(num_cars_to_reset)


class MadrasTrafficAgent(object):
    def __init__(self, port, cfg, name='MadrasTraffic'):
        self.cfg = cfg
        self.name = name
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.env = TorcsEnv(vision=(cfg["vision"] if "vision" in cfg else False),
                            throttle=(cfg["throttle"] if "throttle" in cfg else True),
                            gear_change=(cfg["gear_change"] if "gear_change" in cfg else False),
                            visualise=(self.cfg["visualise"] if "visualise" in self.cfg else False),
                            name=self.name
                           )
        self.PID_controller = PIDController(cfg["pid_settings"])
        self.port = port
        self.min_safe_dist = 0.005*(cfg["min_safe_dist"] if "min_safe_dist" in cfg else 1)  # 1 meter

    def wait_for_observation(self):
        """Refresh client and wait for a valid observation to come in."""
        self.ob = None
        while self.ob is None:
            logging.debug("{} Still waiting for observation at {}".format(self.name, self.port))

            try:
                self.client = snakeoil3.Client(
                    p=self.port,
                    vision=(self.cfg["vision"] if "vision" in self.cfg else False),
                    visualise=(self.cfg["visualise"] if "visualise" in self.cfg else False),
                    name=self.name
                    )
                # Open new UDP in vtorcs
                self.client.MAX_STEPS = self.cfg["client_max_steps"] if "client_max_steps" in self.cfg else np.inf
                self.client.get_servers_input(0)
                # Get the initial input from torcs
                raw_ob = self.client.S.d
                # Get the current full-observation from torcs
                self.ob = self.env.make_observation(raw_ob)
            except:
                pass

    def get_action(self):
        raise NotImplementedError("Successor classes must implement this method")

    def flag_off(self, random_seed=0):
        del random_seed
        self.wait_for_observation()
        logging.debug("[{}]: My server is PID: {}".format(self.name, self.client.serverPID))

        self.is_alive = True
        while True:
            if self.is_alive:
                action = self.get_action()
                try:
                    self.ob, _, done, info = self.env.step(0, self.client, action)
                
                except Exception as e:
                    logging.debug("Exception {} caught by {} traffic agent at port {}".format(
                                  str(e), self.name, self.port))
                    self.wait_for_observation()
                self.detect_and_prevent_imminent_crash_out_of_track()
                self.PID_controller.update(self.ob)
                if done:
                    self.is_alive = False
                    logging.debug("{} died.".format(self.name))


    def get_front_opponents(self):
        return np.array([
            self.ob.opponents[16],
            self.ob.opponents[17],
            self.ob.opponents[18],
            ])

    def detect_and_prevent_imminent_crash_out_of_track(self):
        while True:
            min_dist_from_track = np.min(self.ob.track)
            if min_dist_from_track <= self.min_safe_dist:
                closest_dist_sensor_id = np.argmin(self.ob.track)
                if closest_dist_sensor_id < 9:
                    action = [1, 0, 1]
                elif closest_dist_sensor_id > 9:
                    action = [-1, 0, 1]
                else:
                    action = [0, 0, 1]
                self.ob, _, _, _ = self.env.step(0, self.client, action)
            else:
                break

    def get_collision_cone_radius(self):
        speed = self.ob.speedX * self.env.default_speed * (1000.0 / 3600.0)  # speed in m/sec
        collision_time_window = self.cfg["collision_time_window"] if "collision_time_window" in self.cfg else 1
        collision_cone_radius = collision_time_window * speed
        return collision_cone_radius / 200.0  # Normalizing

    def avoid_impending_collision(self):
        # If the opponent in front is too close, brake
        opponents_in_front = self.get_front_opponents()
        closest_front = np.min(opponents_in_front)
        frontal_distance_threshold = self.get_collision_cone_radius()
        if closest_front < frontal_distance_threshold:
            self.brake = 1
        else:
            self.brake = 0

class ConstVelTrafficAgent(MadrasTrafficAgent):
    def __init__(self, port, cfg, name):
        super(ConstVelTrafficAgent, self).__init__(port, cfg, name)
        self.target_speed = cfg["target_speed"]/self.env.default_speed
        self.target_lane_pos = cfg["target_lane_pos"]

    def get_action(self):
        action = self.PID_controller.get_action([self.target_lane_pos, self.target_speed])
        self.steer, self.accel, self.brake = action[0], action[1], action[2]
        self.avoid_impending_collision()
        return np.asarray([self.steer, self.accel, self.brake])


class SinusoidalSpeedAgent(MadrasTrafficAgent):
    def __init__(self, port, cfg, name):
        super(SinusoidalSpeedAgent, self).__init__(port, cfg, name)
        self.speed_amplitude = self.cfg["speed_amplitude"] / self.env.default_speed
        self.speed_time_period = self.cfg["speed_time_period"]
        self.target_lane_pos = cfg["target_lane_pos"]
        self.time_step = 0

    def get_action(self):
        self.target_speed = self.speed_amplitude * np.sin(self.time_step/self.speed_time_period)
        action = self.PID_controller.get_action([self.target_lane_pos, self.target_speed])
        self.steer, self.accel, self.brake = action[0], action[1], action[2]
        self.avoid_impending_collision()
        self.time_step += 1
        return np.asarray([self.steer, self.accel, self.brake])


class RandomLaneSwitchAgent(MadrasTrafficAgent):
    def __init__(self, port, cfg, name):
        super(RandomLaneSwitchAgent, self).__init__(port, cfg, name)
        self.target_speed = cfg["target_speed"]/self.env.default_speed
        self.time_step = 0
        self.target_lane_pos = 0

    def get_action(self):
        if self.time_step % self.cfg["lane_change_interval"] == 0:
            self.target_lane_pos = 0.75 if np.random.sample() < 0.5 else -0.75
        action = self.PID_controller.get_action([self.target_lane_pos, self.target_speed])
        self.steer, self.accel, self.brake = action[0], action[1], action[2]
        self.avoid_impending_collision()
        return np.asarray([self.steer, self.accel, self.brake])


class RandomStoppingAgent(MadrasTrafficAgent):
    def __init__(self, port, cfg, name):
        super(RandomStoppingAgent, self).__init__(port, cfg, name)
        self.target_speed = cfg["target_speed"]/self.env.default_speed
        self.target_lane_pos = cfg["target_lane_pos"]
        self.max_stop_duration = cfg["max_stop_duration"]
        self.stopped_for = 0

    def get_action(self):
        if self.stopped_for == 0:
            # flip a coin
            if np.random.sample() < 0.1:
                self.stopped_for = np.random.randint(low=0, high=self.max_stop_duration)
        if self.stopped_for == 0:
            action = self.PID_controller.get_action([self.target_lane_pos, self.target_speed])
            self.steer, self.accel, self.brake = action[0], action[1], action[2]
            self.avoid_impending_collision()
        else:
            self.brake = 1
            self.stopped_for -= 1
        return np.asarray([self.steer, self.accel, self.brake])



class ParkedAgent(MadrasTrafficAgent):
    def __init__(self, port, cfg, name):
        super(ParkedAgent, self).__init__(port, cfg, name)
        self.target_speed = cfg["target_speed"]/self.env.default_speed

    def flag_off(self, random_seed=0):
        self.time = time()
        np.random.seed(random_seed)
        parking_lane_pos_range = (self.cfg["parking_lane_pos"]["high"] -
                                  self.cfg["parking_lane_pos"]["low"])
        self.parking_lane_pos = (self.cfg["parking_lane_pos"]["low"] +
                                 np.random.random()*parking_lane_pos_range)
        parking_dist_range = (self.cfg["parking_dist_from_start"]["high"] -
                              self.cfg["parking_dist_from_start"]["low"])
        self.parking_dist_from_start = (self.cfg["parking_dist_from_start"]["low"] +
                                        np.random.random()*parking_dist_range)
        self.behind_finish_line = True
        self.prev_dist_from_start = 0
        self.parked = False
        super(ParkedAgent, self).flag_off()

    def get_action(self):
        self.distance_from_start = self.ob.distFromStart
        if self.distance_from_start + 1 < self.prev_dist_from_start:
            self.behind_finish_line = False
        if not self.behind_finish_line and self.distance_from_start >= self.parking_dist_from_start:
            self.steer, self.accel, self.brake = 0.0, 0.0, 1.0
            if not self.parked:
                logging.debug("{} parked at lanepos: {}, distFromStart: {} after time {} sec".format(
                              self.name, self.parking_lane_pos, self.parking_dist_from_start,
                              time()-self.time))
                self.parked = True
        else:
            action = self.PID_controller.get_action([self.parking_lane_pos, self.target_speed])
            self.steer, self.accel, self.brake = action[0], action[1], action[2]
            self.avoid_impending_collision()
        self.prev_dist_from_start = self.distance_from_start

        return np.asarray([self.steer, self.accel, self.brake])