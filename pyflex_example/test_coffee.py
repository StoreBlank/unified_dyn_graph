import os
import numpy as np
import pyflex
import time


time_step = 500 # 120
# des_dir = 'test_FluidFall'
# os.system('mkdir -p ' + des_dir)

pyflex.init(False)

global_scale = 24

scale = 0.2 * global_scale / 8.0
x = -0.9 * global_scale / 8.0
y = 0.5
z = -0.9 * global_scale / 8.0
staticFriction = 0.0
dynamicFriction = 1.0
draw_skin = 1.0
num_coffee = 1000 # [200, 1000]
scene_params = np.array([
    scale, x, y, z, staticFriction, dynamicFriction, draw_skin, num_coffee])
pyflex.set_scene(20, scene_params, 0)

for i in range(time_step):
    # pyflex.step(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pyflex.step()

pyflex.clean()