"""Traffic agent created using just snakeoil."""

import sys
import numpy as np
import utils.snakeoil3_gym as snakeoil3
from utils.madras_datatypes import Madras

madras = Madras()
PORT = madras.intX(sys.argv[1])
PI = 3.14159265359
maxSteps = 5000


def drive_traffic(c, speed):
    """Drive Function."""
    S, R = c.S.d, c.R.d
    opp = np.asarray(S['opponents'])
    closest = np.min(opp)
    front = np.array([opp[15], opp[16], opp[17], opp[18], opp[19]])
    closest_front = np.min(front)
    print("CLOSEST {}".format(closest))
    print("CLOSEST_FRONT {}".format(closest_front))
    target_speed = madras.floatX(speed)

    # Steer To Corner
    R['steer'] = (S['angle'] + madras.floatX(sys.argv[3]) / 10) * 10 / PI
    # Steer To Center
    R['steer'] -= (S['trackPos'] + madras.floatX(sys.argv[3])) * .10

    # Traction Control System
    if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
       (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
        R['accel'] -= .2

    # Breaking & acceleration
    if closest_front <= (madras.floatX(0.5 * target_speed) + 10.0):
        print("BREAK")
        if S['speedX'] > 0:
            R['gear'] = -1
            R['accel'] = 1
        else:
            R['gear'] = 1
            R['break'] = 1
            R['accel'] = 0

    else:
        R['break'] = 0
        # Automatic Transmission
        R['gear'] = 1
        if S['speedX'] > 50:
            R['gear'] = 2
        if S['speedX'] > 80:
            R['gear'] = 3
        if S['speedX'] > 110:
            R['gear'] = 4
        if S['speedX'] > 140:
            R['gear'] = 5
        if S['speedX'] > 170:
            R['gear'] = 6
        # Throttle Control
        if S['speedX'] < target_speed - (R['steer'] * 50):
            R['accel'] += .01
        else:
            R['accel'] -= .01
        if S['speedX'] < 10:
            R['accel'] += 1 / (S['speedX'] + .1)

    print(S['speedX'], R['gear'])
    return


if __name__ == "__main__":
    C = snakeoil3.Client(p=PORT, e=maxSteps, vision=False)
    print(sys.argv[1])
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input(0)
        drive_traffic(C, sys.argv[1])
        C.respond_to_server()
    C.shutdown()
