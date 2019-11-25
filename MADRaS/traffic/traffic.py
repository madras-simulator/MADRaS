import numpy as np
from utils.gym_torcs import TorcsEnv
import utils.snakeoil3_gym as snakeoil3
from controllers.pid import PIDController
import utils.madras_datatypes as md
from multiprocessing import Process

MadrasDatatypes = md.MadrasDatatypes()

class MadrasTrafficManager(object):
    """Creates the traffic agents for a given training configuration."""
    def __init__(self, cfg):
        self.traffic_agents = {}
        self.traffic_processes = []
        for i, agent in enumerate(cfg):
            agent_type = [x for x in agent.keys()][0]  # TODO(santara): find a better way of extracting key from a dictionary with a single entry
            agent_config = agent[agent_type]
            agent_port = agent_config["port"]
            try:
                import pdb; pdb.set_trace()
                exec("self.traffic_agents['{}_{}'] = {}({}, {})".format(
                    agent_type, i, agent_type, agent_port, agent_config))
            except Exception as e:
                raise ValueError("Unknown traffic class {} or incomplete specification.\n Original error: {}".format(
                                 agent_type, e))
    
    def get_traffic_agents(self):
        return self.traffic_agents

    def flag_off_traffic(self):
        for agent in self.traffic_agents.values():
            self.traffic_processes.append(Process(target=agent.flag_off))
        for traffic_process in self.traffic_processes:
            traffic_process.start()

    def kill_all_traffic_agents(self):
        for traffic_process in self.traffic_processes:
            traffic_process.terminate()
        self.traffic_processes = []

    def reset(self):
        self.kill_all_traffic_agents()
        self.flag_off_traffic()


class MadrasTrafficAgent(object):
    def __init__(self, port, cfg):
        self.cfg = cfg
        self.type = None
        self.env = TorcsEnv(vision=(cfg["vision"] if "vision" in cfg else False),
                            throttle=(cfg["throttle"] if "throttle" in cfg else True),
                            gear_change=(cfg["gear_change"] if "gear_change" in cfg else False),
                            visualise=(self.cfg["visualise"] if "visualise" in self.cfg else False)
                           )
        self.PID_controller = PIDController(cfg["pid_settings"])
        self.port = port

    def wait_for_observation(self):
        """Refresh client and wait for a valid observation to come in."""
        self.ob = None
        while self.ob is None:
            try:
                self.client = snakeoil3.Client(p=self.port,
                                               vision=(self.cfg["vision"] if "vision" in self.cfg else False),
                                               visualise=(self.cfg["visualise"] if "visualise" in self.cfg else False)
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

    def flag_off(self, sleep=0):
        self.wait_for_observation()
        self.is_alive = True
        while True:
            action = self.get_action()
            if self.is_alive:
                try:
                    self.ob, _, done, info = self.env.step(0, self.client, action)
                
                except Exception as e:
                    print("Exception {} caught by {} traffic agent at port {}".format(
                            str(e), self.type, self.port))
                    self.wait_for_observation()
                if done:
                    self.is_alive = False


class ConstVelTrafficAgent(MadrasTrafficAgent):
    def __init__(self, port, cfg):
        super(ConstVelTrafficAgent, self).__init__(port, cfg)
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.target_speed = cfg["target_speed"]/self.env.default_speed
        self.target_lane_pos = cfg["target_lane_pos"]

    def get_front_opponents(self):
        return np.array([self.ob.opponents[15],
                         self.ob.opponents[16],
                         self.ob.opponents[17],
                         self.ob.opponents[18],
                         self.ob.opponents[19]])

    def get_action(self):
        action = self.PID_controller.get_action([self.target_speed, self.target_lane_pos])
        self.steer, self.accel, self.brake = action[0], action[1], action[2]

        # If the opponent in front is too close, brake
        opponents_in_front = self.get_front_opponents()
        closest_front = np.min(opponents_in_front)
        frontal_distance_threshold = MadrasDatatypes.floatX(((0.5 * self.ob.speedX * 100) + 10.0) / 200.0)
        if closest_front < frontal_distance_threshold:
            self.brake = 1
        else:
            self.brake = 0
        return np.asarray([self.steer, self.accel, self.brake])


