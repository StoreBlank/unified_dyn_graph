import os
import numpy as np
import pyflex
import time

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)

time_step = 200 # 120

pyflex.init(False)

global_scale = 12

x = 0.
y = 1.
z = 0.
length = 2.0
stiffness = 1.2
draw_mesh = 1

scene_params = np.array([x, y, z, length, stiffness, draw_mesh])\

pyflex.set_scene(25, scene_params, 0) 

pyflex.set_camPos(np.array([0.13, 2.0, 3.2]))

for j in range(time_step):
    pyflex.step()
pyflex.clean()
