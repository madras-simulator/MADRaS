"""Sinusoidally speed changing Traffic Agent."""
import sys
import math
import numpy as np
from controllers.pid import PID
from utils.gym_torcs import TorcsEnv
import utils.snakeoil3_gym as snakeoil3
from utils.madras_datatypes import Madras

madras = Madras()

def playTraffic(port=3101, target_vel=50.0, angle=0.0, sleep=0):
    """Traffic Play function."""
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    client = snakeoil3.Client(p=port, vision=False)
    client.get_servers_input(step=0)
    obs = client.S.d
    ob = env.make_observation(obs)
    episode_count = max_eps
    max_steps = max_steps_eps_traffic
    early_stop = 0
    velocity_i = target_vel / 300.0
    accel_pid = PID(np.array([10.5, 0.05, 2.8]))
    steer_pid = PID(np.array([5.1, 0.001, 0.000001]))
    steer = 0.0
    accel = 0.0
    brake = 0
    A = amplitude / 300.0
    T = time_period
    print velocity_i
    for i in range(episode_count):
        info = {'termination_cause': 0}
        steer = 0.0
        accel = 0.0
        brake = 0
        S = 0.0
        for step in range(max_steps):
            a_t = np.asarray([steer, accel, brake])  # [steer, accel, brake]
            ob, r_t, done, info = env.step(step, client, a_t, early_stop, 1)
            if done:
                break
            if (step <= sleep):
                print "WAIT"
                continue
            velocity = velocity_i + A * math.sin(step / T)
            opp = ob.opponents
            front = np.array([opp[15], opp[16], opp[17], opp[18], opp[19]])
            closest_front = np.min(front)
            if (step % T == 0):
                print("VEL_CHANGE: ", velocity * 300.0)
            vel_error = velocity - ob.speedX
            angle_error = -(ob.trackPos - angle) / 10 + ob.angle
            steer_pid.update_error(angle_error)
            accel_pid.update_error(vel_error)
            accel = accel_pid.output()
            steer = steer_pid.output()
            if accel < 0:
                brake = 1
            else:
                brake = 0
            if closest_front < ((float(0.5 * ob.speedX * 100) + 10.0) / 200.0):
                brake = 1
            else:
                brake = 0

        if 'termination_cause' in info.keys() and info['termination_cause'] == 'hardReset':
            print 'Hard reset by some agent'
            ob, client = env.reset(client=client)

if __name__ == "__main__":

    try:
        port = madras.intX(sys.argv[1])
    except Exception as e:
        # raise e
        print("Usage : python %s <port>" % (sys.argv[0]))
        sys.exit()

    playTraffic(port=port, target_vel=madras.floatX(sys.argv[2]),
                angle=madras.floatX(sys.argv[3]), sleep=madras.intX(sys.argv[4]))
