import threading
import multiprocessing
import traffic.const_vel as playGame_const_vel
import traffic.const_vel as playGame_const_vel_s

num_workers = 3 
print("numb of workers is" + str(num_workers))
for i in range(num_workers):
      worker_work = lambda: (playGame_const_vel.playTraffic(port=3001+i))

for i in range(num_workers):
      worker_work2 = lambda: (playGame_const_vel_s.playTraffic(port=3001+i+num_workers))

for i in range(num_workers):
      t = threading.Thread(target=(worker_work))
      t2 = threading.Thread(target=(worker_work2))
      t.start()
      t2.start()
