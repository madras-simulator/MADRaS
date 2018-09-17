"""
Gym Madras Env Wrapper.

This is an OpenAI gym environment wrapper for the MADRaS simulator. For more information on the OpenAI Gym interface, please refer to: https://gym.openai.com

Built on top of gym_torcs https://github.com/ugo-nama-kun/gym_torcs/blob/master/gym_torcs.py

The following enhancements were made for Multi-agent synchronization using exception handling:
- All the agents connect to the same TORCS engine through UDP ports
- If an agent fails to connect to the TORCS engine, it keeps on trying in a loop until successful
- Restart the episode for all the agents when any one of the learning agents terminates its episode

"""
import sys
sys.path.append('../../utils/')
sys.path.append('../../controllers')
import math
import yaml
from copy import deepcopy
import numpy as np
import snakeoil3_gym as snakeoil3
from gym_torcs import TorcsEnv
from pid import PID

with open("./configurations.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)


class MadrasEnv(TorcsEnv):
    """Definition of the Gym Madras Env."""
    def __init__(self, vision=False, throttle=True,
                 gear_change=False, port=3001, pid_assist=True,
                 CLIENT_MAX_STEPS=np.inf):
        """Init Method."""
        self.pid_assist = pid_assist
        if self.pid_assist:
            self.action_dim = 2  # LanePos, Velocity
        else:
            self.action_dim = 3  # Accel, Steer, Brake
        TorcsEnv.__init__(self, vision=False, throttle=True, gear_change=False)
        self.state_dim = 29  # No. of sensors input
        self.env_name = 'Madras_Env'
        self.port = port
        self.CLIENT_MAX_STEPS = CLIENT_MAX_STEPS
        self.client_type = 0  # Snakeoil client type
        self.initial_reset = True
        self.early_stop = True
        if self.pid_assist:
            self.PID_latency = 10
        else:
            self.PID_latency = 1
        self.accel_PID = PID(np.array([10.5, 0.05, 2.8]))  # accel PID
        self.steer_PID = PID(np.array([5.1, 0.001, 0.000001]))  # steer PID

        self.prev_lane = 0
        self.prev_angle = 0
        self.prev_vel = 0
        self.prev_dist = 0
        self.ob = None

    def reset(self, prev_step_info=None):
        """Reset Method. To be called at the end of each episode"""
        if self.initial_reset:
            while self.ob is None:
                try:
                    self.client = snakeoil3.Client(p=self.port,
                                                   vision=self.vision)
                    # Open new UDP in vtorcs
                    self.client.MAX_STEPS = self.CLIENT_MAX_STEPS
                    self.client.get_servers_input(step=0)
                    # Get the initial input from torcs
                    raw_ob = self.client.S.d
                    # Get the current full-observation
                    self.ob = self.make_observation(raw_ob)
                except:
                    pass
            self.initial_reset = False

        else:
            try:
                if 'termination_cause' in list(prev_step_info.keys()) and\
                        prev_step_info['termination_cause'] == 'hardReset':
                    self.ob, self.client =\
                        TorcsEnv.reset(self, client=self.client, relaunch=True)
                else:
                    self.ob, self.client =\
                        TorcsEnv.reset(self, client=self.client, relaunch=True)

            except Exception as e:
                self.ob = None
                while self.ob is None:
                    try:
                        self.client = snakeoil3.Client(p=self.port,
                                                       vision=self.vision)
                        # Open new UDP in vtorcs
                        self.client.MAX_STEPS = self.CLIENT_MAX_STEPS
                        self.client.get_servers_input(step=0)
                        # Get the initial input from torcs
                        raw_ob = self.client.S.d
                        # Get the current full-observation from torcs
                        self.ob = self.make_observation(raw_ob)
                    except:
                        pass

        self.distance_traversed = 0
        s_t = np.hstack((self.ob.angle, self.ob.track, self.ob.trackPos,
                        self.ob.speedX, self.ob.speedY, self.ob.speedZ,
                        self.ob.wheelSpinVel / 100.0, self.ob.rpm))

        return s_t

    def step(self, desire):
        """Step method to be called at each time step."""
        r_t = 0

        for PID_step in range(self.PID_latency):
                # Implement the desired trackpos and velocity using PID
            if self.pid_assist:
                self.accel_PID.update_error((desire[1] - self.prev_vel))
                self.steer_PID.update_error((-(self.prev_lane - desire[0]) / 10 +
                                            self.prev_angle))
                if self.accel_PID.output() < 0.0:
                    brake = 1
                else:
                    brake = 0
                a_t = np.asarray([self.steer_PID.output(),
                                 self.accel_PID.output(), brake])
            else:
                a_t = desire
            try:
                self.ob, r, done, info = TorcsEnv.step(self, PID_step,
                                                       self.client, a_t,
                                                       self.early_stop)
            except Exception as e:
                print(("Exception caught at port " + str(e)))
                self.ob = None
                while self.ob is None:
                    try:
                        self.client = snakeoil3.Client(p=self.port,
                                                       vision=self.vision)
                        # Open new UDP in vtorcs
                        self.client.MAX_STEPS = self.CLIENT_MAX_STEPS
                        self.client.get_servers_input(0)
                        # Get the initial input from torcs
                        raw_ob = self.client.S.d
                        # Get the current full-observation from torcs
                        self.ob = self.make_observation(raw_ob)
                    except:
                        pass
                    continue
            self.prev_vel = self.ob.speedX
            self.prev_angle = self.ob.angle
            self.prev_lane = self.ob.trackPos
            if (math.isnan(r)):
                r = 0.0
            r_t += r  # accumulate rewards over all the time steps

            self.distance_traversed = self.client.S.d['distRaced']
            r_t += (self.distance_traversed - self.prev_dist) /\
                cfg['madras']['track_len']
            self.prev_dist = deepcopy(self.distance_traversed)
            if self.distance_traversed >= cfg['madras']['track_len']:
                done = True
            if done:
                break

        s_t1 = np.hstack((self.ob.angle, self.ob.track, self.ob.trackPos,
                          self.ob.speedX, self.ob.speedY, self.ob.speedZ,
                          self.ob.wheelSpinVel / 100.0, self.ob.rpm))

        return s_t1, r_t, done, info
