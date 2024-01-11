import os
import numpy as np
import pyflex
import time
import argparse

from utils_env import rand_float

pyflex.init(False)

x = rand_float(8.0, 10.0)
y = rand_float(8.0, 10.0)
z = rand_float(8.0, 10.0)
clusterStiffness = rand_float(0.3, 0.7)
# clusterPlasticThreshold = rand_float(0.000004, 0.0001)
clusterPlasticThreshold = rand_float(0.00001, 0.0005)
clusterPlasticCreep = rand_float(0.1, 0.3)
scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])

temp = np.array([0])
pyflex.set_scene(5, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

for _ in range(200):
    pyflex.step()