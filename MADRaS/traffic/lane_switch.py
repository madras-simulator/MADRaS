"""Random Lane changing Traffic Agent."""
import sys
import time
import yaml
import random
import numpy as np
from controllers.pid import PID
from utils.gym_torcs import TorcsEnv
import utils.snakeoil3_gym as snakeoil3
from utils.madras_datatypes import Madras

madras = Madras()
with open("./traffic/configurations.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

random.seed(time.time())


def playTraffic(port=3101, target_vel=50.0, angle=0.0, sleep=0):
    """Traffic Play function."""
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    ob = None
    while ob is None:
        try:
            client = snakeoil3.Client(p=port, vision=False)
            client.MAX_STEPS = np.inf
            client.get_servers_input(step=0)
            obs = client.S.d
            ob = env.make_observation(obs)
        except:
            pass
    episode_count = cfg['traffic']['max_eps']
    max_steps = cfg['traffic']['max_steps_eps']
    early_stop = 0
    velocity = target_vel / 300.0
    accel_pid = PID(np.array([10.5, 0.05, 2.8]))
    steer_pid = PID(np.array([5.1, 0.001, 0.000001]))
    steer = 0.0
    accel = 0.0
    brake = 0
    print(velocity)
    for i in range(episode_count):
        info = {'termination_cause': 0}
        steer = 0.0
        accel = 0.0
        brake = 0

        for step in range(max_steps):
            a_t = np.asarray([steer, accel, brake])  # [steer, accel, brake]
            try:
                ob, r_t, done, info = env.step(step, client, a_t, early_stop)
                if done:
                    break
            except Exception as e:
                print("Exception caught at port " + str(i) + str(e))
                ob = None
                while ob is None:
                    try:
                        client = snakeoil3.Client(p=port, vision=False)
                        client.MAX_STEPS = np.inf
                        client.get_servers_input(step=0)
                        obs = client.S.d
                        ob = env.make_observation(obs)
                    except:
                        pass
                    continue
            if (step <= sleep):
                print("WAIT" + str(port))
                continue
            opp = ob.opponents
            front = np.array([opp[15], opp[16], opp[17], opp[18], opp[19]])
            closest_front = np.min(front)
            print(ob.speedX * 300)
            vel_error = velocity - ob.speedX
            angle_error = -(ob.trackPos + angle) / 10 + ob.angle + random.choice([1, -1]) * 0.05
            steer_pid.update_error(angle_error)
            accel_pid.update_error(vel_error)
            accel = accel_pid.output()
            steer = steer_pid.output()
            if accel < 0:
                brake = 1
            else:
                brake = 0
            if closest_front < ((madras.floatX(0.5 * ob.speedX * 100) + 10.0) / 200.0):
                brake = 1
            else:
                brake = 0

        try:
            if 'termination_cause' in info.keys() and info['termination_cause'] == 'hardReset':
                print('Hard reset by some agent')
                ob, client = env.reset(client=client, relaunch=True)

        except Exception as e:
            print("Exception caught at point B at port " + str(i) + str(e) )
            ob = None
            while ob is None:
                try:
                    client = snakeoil3.Client(p=port, vision=False)  # Open new UDP in vtorcs
                    client.MAX_STEPS = np.inf
                    client.get_servers_input(0)  # Get the initial input from torcs
                    obs = client.S.d  # Get the current full-observation from torcs
                    ob = env.make_observation(obs)
                except:
                    print("Exception caught at at point C at port " + str(i) + str(e))

if __name__ == "__main__":

    try:
        port = madras.intX(sys.argv[1])
    except Exception as e:
        # raise e
        print("Usage : python %s <port>" % (sys.argv[0]))
        sys.exit()

    playTraffic(port=port, target_vel=madras.floatX(sys.argv[2]),
                angle=madras.floatX(sys.argv[3]), sleep=madras.intX(sys.argv[4]))
