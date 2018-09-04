#!/usr/bin/python

import numpy as np
import snakeoil3_gym as snakeoil3
import sys

PORT = int(sys.argv[2])
PI= 3.14159265359
maxSteps = 5000

def drive_traffic(c, speed):
    S,R= c.S.d,c.R.d
    opp = np.asarray(S[u'opponents'])
    closest = np.min(opp)
    front = np.array([opp[15],opp[16],opp[17],opp[18],opp[19]])
    closest_front = np.min(front)
    print "CLOSEST {}".format(closest)
    print "CLOSEST_FRONT {}".format(closest_front)
    index = np.argmin(opp)
    #print index
    #print opp
    #print "17: {}  18: {} 19: {}".format(opp[16],opp[17],opp[18])
    target_speed= float(speed)

    # Steer To Corner
    R[u'steer']= (S[u'angle'] + float(sys.argv[3])/10)*10 / PI
    # Steer To Center
    R[u'steer']-= (S[u'trackPos'] + float(sys.argv[3]))*.10


    # Traction Control System
    if ((S[u'wheelSpinVel'][2]+S[u'wheelSpinVel'][3]) -
       (S[u'wheelSpinVel'][0]+S[u'wheelSpinVel'][1]) > 5):
       R[u'accel']-= .2


    # Breaking & acceleration
    if closest_front<=(float(0.5*target_speed) + 10.0):
        print "BREAK"
        if S[u'speedX'] > 0:
            R[u'gear']=-1
            R[u'accel'] = 1
        else :
            R[u'gear'] = 1
            R[u'break'] = 1
            R[u'accel'] = 0

    else :
        R[u'break']=0
        # Automatic Transmission
        R[u'gear']=1
        if S[u'speedX']>50:
            R[u'gear']=2
        if S[u'speedX']>80:
            R[u'gear']=3
        if S[u'speedX']>110:
            R[u'gear']=4
        if S[u'speedX']>140:
            R[u'gear']=5
        if S[u'speedX']>170:
            R[u'gear']=6
        # Throttle Control
        if S[u'speedX'] < target_speed - (R[u'steer']*50):
            R[u'accel']+= .01
        else:
            R[u'accel']-= .01
        if S[u'speedX']<10:
            R[u'accel']+= 1/(S[u'speedX']+.1)

    print S[u'speedX'],R[u'gear']
    return


if __name__ == u"__main__":
	C = snakeoil3.Client(p=PORT, e=maxSteps, vision=False)
	print sys.argv[1]
	for step in xrange(C.maxSteps, 0, -1):
		C.get_servers_input(0)
		drive_traffic(C, sys.argv[1])
		C.respond_to_server()
	C.shutdown()
