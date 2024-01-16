import time
import numpy as np
import cv2

import pygame
from box_sim import BoxSim

"""
Minimal example for pile simulation.
"""


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

sim = BoxSim()
count = 0

scale = 10.
lo = 50.
hi = 450.
x = rand_float(lo, hi)
y = rand_float(lo, hi)
# x = 250.
# y = 250.
dx = 0.
dy = 0.

# allow the simulator to a resting position
n_iter_rest = 2
for i in range(n_iter_rest):
    sim.update((x, y))

n_sim_step = 50
for i in range(n_sim_step):

    print("%d/%d" % (i, n_sim_step))

    x = x + dx
    y = y + dy
    x = np.clip(x, lo, hi)
    y = np.clip(y, lo, hi)

    dx = dx + rand_float(-scale, scale) - (x - 250.) * scale * 0.01
    dy = dy + rand_float(-scale, scale) - (y - 250.) * scale * 0.01


    sim.update((x, y))

    time.sleep(0.1)