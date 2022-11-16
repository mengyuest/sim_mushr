from re import S
import range_libc
import numpy as np
import itertools, time
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt

####################################################################################################
#
#                                              WARNING
#  
#
#                    This file uses range_libc in it's native coordinate space.
#                      Use this method at your own peril since the coordinate
#                      conversions are nontrivial from ROS's coordinate space.
#                       Ignore this warning if you intend to use range_libc's 
#                                   left handed coordinate space
#
#
####################################################################################################

# print range_libc.USE_CACHED_TRIG
# print range_libc.USE_CACHED_TRIG
# print range_libc.USE_ALTERNATE_MOD
# print range_libc.USE_CACHED_CONSTANTS
# print range_libc.USE_FAST_ROUND
# print range_libc.NO_INLINE
# print range_libc.USE_LRU_CACHE
# print range_libc.LRU_CACHE_SIZE

class FakeLidar:
	def __init__(self):
		# basic parameters
		self.THETA_DISCRETIZATION = 720.0
		self.MIN_RANGE_METERS = 0.02
		self.MAX_RANGE_METERS = 10.0
		# self.ANGLE_STEP = 0.86*np.pi/180.0  # in radians, at max freq 12Hz
		self.ANGLE_STEP = 0.5*np.pi/180.0  # in radians, at freq 7hz
		# self.ANGLE_STEP = 20*np.pi/180.0  # in radians, at freq 7hz
		self.ANGLE_MIN = -np.pi
		self.ANGLE_MAX = np.pi
		self.ANGLES = np.arange(self.ANGLE_MIN, self.ANGLE_MAX, self.ANGLE_STEP, dtype=np.float32)
		self.CAR_LENGTH = 0.33
		self.Z_SHORT = 0.16
		self.Z_MAX = 0.16
		self.Z_BLACKOUT_MAX = 50
		self.Z_RAND = 0.01
		self.Z_HIT = 0.8
		self.Z_SIGMA = 0.03

		self.map_resolution = 20.0/300.0 # meters per cell (about 20m width for the regular house, and 300 pixels per image)
		
		# map_name = b"/home/azureuser/hackathon_data/house_expo/png/0004d52d1aeeb8ae6de39d6bd993e992.png"
		map_name = b"../maps/synthetic.map.png"
		occ_map = range_libc.PyOMap(map_name, 1)
		max_range_px = int(self.MAX_RANGE_METERS / self.map_resolution)

		self.range_method = range_libc.PyBresenhamsLine(occ_map, max_range_px)
        # position, orientation = self.tl.lookupTransform(
        #     self.TF_PREFIX + "base_link", self.TF_PREFIX + "laser_link", rospy.Time(0)
        # )
        # self.x_offset = position[0]

	def noise_laser_scan(self, ranges):
		indices = np.zeros(ranges.shape[0], dtype=np.int)
		prob_sum = self.Z_HIT + self.Z_RAND + self.Z_SHORT
		hit_count = int((self.Z_HIT / prob_sum) * indices.shape[0])
		rand_count = int((self.Z_RAND / prob_sum) * indices.shape[0])
		short_count = indices.shape[0] - hit_count - rand_count
		indices[hit_count : hit_count + rand_count] = 1
		indices[hit_count + rand_count :] = 2
		np.random.shuffle(indices)

		hit_indices = indices == 0
		ranges[hit_indices] += np.random.normal(
	        loc=0.0, scale=self.Z_SIGMA, size=hit_count
	    )[:]

		rand_indices = indices == 1
		ranges[rand_indices] = np.random.uniform(
	        low=self.MIN_RANGE_METERS, high=self.MAX_RANGE_METERS, size=rand_count
	    )[:]

		short_indices = indices == 2
		ranges[short_indices] = np.random.uniform(
	        low=self.MIN_RANGE_METERS, high=ranges[short_indices], size=short_count
	    )[:]

		max_count = (self.Z_MAX / (prob_sum + self.Z_MAX)) * ranges.shape[0]
		while max_count > 0:
			cur = np.random.randint(low=0, high=ranges.shape[0], size=1)
			blackout_count = np.random.randint(low=1, high=self.Z_BLACKOUT_MAX, size=1)
			while (
	            cur > 0
	            and cur < ranges.shape[0]
	            and blackout_count > 0
	            and max_count > 0
	        ):
				if not np.isnan(ranges[cur]):
					ranges[cur] = np.nan
					cur += 1
					blackout_count -= 1
					max_count -= 1
				else:
					break

	def calc_returns(self):
		laser_angle = np.pi/2.0
		laser_pose_x = 510
		laser_pose_y = 520 
		ranges = np.zeros(len(self.ANGLES) * 1, dtype=np.float32)
		range_pose = np.array((laser_pose_x, laser_pose_y, laser_angle), dtype=np.float32).reshape(1, 3)
		self.range_method.calc_range_repeat_angles(range_pose, self.ANGLES, ranges)
		self.noise_laser_scan(ranges)
		return ranges
	
	def print_image(self, file_name):
		self.range_method.saveTrace(str.encode(file_name))


lidar_obj = FakeLidar()
lidar_obj.calc_returns()
lidar_obj.print_image("../maps/synthetic.map.png")

# testMap = range_libc.PyOMap("../maps/basement_hallways_5cm.png",1)
testMap = range_libc.PyOMap(b"../maps/synthetic.map.png", 1)
# testMap = range_libc.PyOMap("/home/racecar/racecar-ws/src/TA_examples/lab5/maps/basement.png",1)

# TODO
map_arr = np.ones((100, 102))
map_arr[1:98,1:98] = 0
for i in range(50):
	map_arr[i, 2*i+1] = 1
	map_arr[i, 2*i] = 1
	map_arr[i, 2*i-1] = 1
testMap = range_libc.PyOMap(map_arr)
print("FLAG")


if testMap.error():
	exit()
# testMap.save("./test.png")

num_vals = 100000

vals = np.zeros((3,num_vals), dtype=np.float32)
vals[0,:] = testMap.width()/2.0
vals[1,:] = testMap.height()/2.0
vals[2,:] = np.linspace(0,2.0*np.pi, num=num_vals)

def make_scan(x,y,theta,n_ranges):
	MAX_SCAN_ANGLE = (np.pi)
	bl = range_libc.PyBresenhamsLine(testMap, 300)
	# bl = range_libc.PyRayMarching(testMap, 500)
	queries = np.zeros((n_ranges,3),dtype=np.float32)
	ranges = np.zeros(n_ranges,dtype=np.float32)
	queries[:,0] = x
	queries[:,1] = y
	queries[:,2] = theta + np.linspace(-MAX_SCAN_ANGLE, MAX_SCAN_ANGLE, n_ranges)
	bl.calc_range_many(queries,ranges)
	bl.saveTrace(b"test_bla.png")

# make_scan(510,520,np.pi/2.0,720)
make_scan(10,30,np.pi/2.0,720)


















# print("Init: bl")
# bl = range_libc.PyBresenhamsLine(testMap, 500)
# print ("Init: rm")
# rm = range_libc.PyRayMarching(testMap, 500)
# print ("Init: cddt")
# cddt = range_libc.PyCDDTCast(testMap, 500, 108)
# cddt.prune()
# print ("Init: glt")
# glt = range_libc.PyGiantLUTCast(testMap, 500, 108)

# this is for testing the amount of raw functional call overhead, does not compute ranges
# null = range_libc.PyNull(testMap, 500, 108)

# for x in range(10):
# 	vals = np.random.random((3,num_vals)).astype(np.float32)
# 	vals[0,:] *= (testMap.width() - 2.0)
# 	vals[1,:] *= (testMap.height() - 2.0)
# 	vals[0,:] += 1.0
# 	vals[1,:] += 1.0
# 	vals[2,:] *= np.pi * 2.0
# 	ranges = np.zeros(num_vals, dtype=np.float32)

# 	test_states = [None]*num_vals
# 	for i in range(num_vals):
# 		test_states[i] = (vals[0,i], vals[1,i], vals[2,i])

# 	def bench(obj,name):
# 		print ("Running:", name)
# 		start = time.clock()
# 		obj.calc_range_many(vals, ranges)
# 		end = time.clock()
# 		dur_np = end - start
# 		print (",,,"+name+" np: finished computing", ranges.shape[0], "ranges in", dur_np, "sec")
# 		start = time.clock()
# 		ranges_slow = list(map(lambda x: obj.calc_range(*x), test_states))
# 		end = time.clock()
# 		dur = end - start

# 		diff = np.linalg.norm(ranges - np.array(ranges_slow))
# 		if diff > 0.001:
# 			print (",,,"+"Numpy result different from slow result, investigation possibly required. norm:", diff)
# 		# print "DIFF:", diff

# 		print (",,,"+name+": finished computing", ranges.shape[0], "ranges in", dur, "sec")
# 		print (",,,"+"Numpy speedup:", dur/dur_np)

# 	bench(bl, "bl")
# 	bench(rm, "rm")
# 	bench(cddt, "cddt")
# 	bench(glt, "glt")

	# ranges_bl = np.zeros(num_vals, dtype=np.float32)
	# ranges_rm = np.zeros(num_vals, dtype=np.float32)
	# ranges_cddt = np.zeros(num_vals, dtype=np.float32)
	# ranges_glt = np.zeros(num_vals, dtype=np.float32)

	# bl.calc_range_np(vals, ranges_bl)
	# rm.calc_range_np(vals, ranges_rm)
	# cddt.calc_range_np(vals, ranges_cddt)
	# glt.calc_range_np(vals, ranges_glt)

	# diff = ranges_rm - ranges_cddt
	# norm = np.linalg.norm(diff)
	# avg = np.mean(diff)
	# min_v = np.min(diff)
	# max_v = np.max(diff)
	# median = np.median(diff)
	# print avg, min_v, max_v, median

	# plt.hist(diff, bins=1000, normed=1, facecolor='green', alpha=0.75)
	# plt.show()

	# this is for testing the amount of raw functional call overhead, does not compute ranges
	# bench(null, "null")
print ("DONE")