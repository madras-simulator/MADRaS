import sys
sys.path.append('./sample_DDPG_agent/')
import numpy as np
np.random.seed(1337)
from gym_torcs import TorcsEnv
import snakeoil3_gym as snakeoil3
from pid import PID
from configurations import *
import time
import random
random.seed(time.time())


def playTraffic(port=3101, target_vel=50.0, angle=0.0, sleep=0):
	env = TorcsEnv(vision=False, throttle=True, gear_change=False)
	client = snakeoil3.Client(p=port, vision=False)
	client.get_servers_input(step=0,client_type=1)
	obs = client.S.d
	ob = env.make_observation(obs)
	accel_pid = PID(np.array([10.5,0.05,2.8]))
	steer_pid = PID(np.array([5.1,0.001,0.000001]))
	episode_count = max_eps
	max_steps = max_steps_eps_traffic
	early_stop = 0
	velocity_i = target_vel/300.0
	T = random_stop_steps
	steer = 0.0
	accel = 0.0
	brake = 0
	for i in range(episode_count):
		info = {'termination_cause':0}
		steer = 0.0
		accel = 0.0
		brake = 0
		S = 0.0
		initiate_step = 0
		trigger = 0
		velocity = velocity_i
		for step in xrange(max_steps):
			a_t = np.asarray([steer, accel, brake]) # [steer, accel, brake]
			ob, r_t, done, info = env.step(step, client, a_t, early_stop, 1)
			if done:
				break
			if (step <= sleep):
				print "WAIT"
				continue
			if (random.randint(0,1) == 1 and trigger == 0 and step%500 ==0):
				trigger = 1
				initiate_step = step
				velocity = 0
				brake = 1
				print "STOP: ",step
			if (step == (initiate_step + T)):
				velocity = velocity_i
				trigger = 0
				brake = 0
				print "START"
			opp = ob.opponents
			front = np.array([opp[15],opp[16],opp[17],opp[18],opp[19]])
			closest_front = np.min(front)
			vel_error = velocity - ob.speedX
			angle_error = -(ob.trackPos-angle)/10 + ob.angle
			steer_pid.update_error(angle_error)
			accel_pid.update_error(vel_error)
			accel = accel_pid.output()
			steer = steer_pid.output()
			if accel < 0:
				brake = 1
			else :
				brake = 0
			if closest_front < ((float(0.5*ob.speedX*100) + 10.0)/200.0):
				brake = 1
			else :
				brake = 0

		if 'termination_cause' in info.keys() and info['termination_cause']=='hardReset':
			print 'Hard reset by some agent'
			ob, client = env.reset(client=client, client_type=1)

if __name__ == "__main__":

	try:
		port = int(sys.argv[1])
	except Exception as e:
		# raise e
		print "Usage : python %s <port>" % (sys.argv[0])
		sys.exit()

	playTraffic(port=port,target_vel=float(sys.argv[2]),angle=float(sys.argv[3]),sleep=int(sys.argv[4]))
