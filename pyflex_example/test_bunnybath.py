import os
import numpy as np
import pyflex
import time


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


time_step = 150

pyflex.init(False)

scene_params = np.array([])
# pyflex.set_scene(1, scene_params, 0)
temp = np.array([0])
pyflex.set_scene(1, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

print("Scene Upper:", pyflex.get_scene_upper())
print("Scene Lower:", pyflex.get_scene_lower())

for i in range(time_step):
    # pyflex.step(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pyflex.step()

pyflex.clean()
