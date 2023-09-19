import os
import numpy as np
import pyflex
import time

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

time_step = 300 # 120
# des_dir = 'test_FluidFall'
# os.system('mkdir -p ' + des_dir)

pyflex.init(False)



global_scale = 12

n_instance = 3
dynamic_friction = 0.1
gravity = -9.8
restitution = 0.1
low_bound = 0.09
draw_mesh = True
scale = 0.2 * global_scale / 8.0

scene_params = np.zeros(n_instance * 3 + 4)
scene_params[0] = n_instance
scene_params[1] = gravity
scene_params[2] = scale

for i in range(n_instance):
    x = rand_float(0., 0.1)
    y = rand_float(low_bound, low_bound + 0.01)
    z = rand_float(0., 0.1)
    
    scene_params[3*i + 3] = x
    scene_params[3*i + 4] = y
    scene_params[3*i + 5] = z
    
    low_bound += 0.21

if draw_mesh:
    scene_params[n_instance*3 + 3] = 1 # draw mesh

# pyflex.set_scene(3, scene_params, 0) # 3: RigidFall
pyflex.set_scene(24, scene_params, 0) # 24, 25

## Light setting
screenWidth = 720
screenHeight = 720
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 2.0, 0.1]))
pyflex.set_light_fov(70.)

# camera setting
cam_idx = 0
cam_height = 2.0 * global_scale / 8.0
rad = np.deg2rad(cam_idx)
cam_dis = 0.0 * global_scale / 8.0

camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
camAngle = np.array([rad, -np.deg2rad(90.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

for j in range(time_step):
    # pyflex.step(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pyflex.step()

pyflex.clean()
