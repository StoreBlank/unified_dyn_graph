import os
import numpy as np
import pyflex
import time


time_step = 500 # 120
# des_dir = 'test_FluidFall'
# os.system('mkdir -p ' + des_dir)

pyflex.init(False)

global_scale = 15

scale = 0.2 * global_scale / 8.0
x = -0.9 * global_scale / 8.0
y = 0.5
z = -0.9 * global_scale / 8.0
staticFriction = 0.0
dynamicFriction = 1.0
draw_skin = 1.0
num_capsule = 500 # [200, 1000]
slices = 10
segments = 20
scene_params = np.array([scale, x, y, z, staticFriction, dynamicFriction, draw_skin, num_capsule, slices, segments])

temp = np.array([0])
pyflex.set_scene(21, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

## Light setting
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

folder_dir = '../ptcl_data/capsule'
os.system('mkdir -p ' + folder_dir)

des_dir = folder_dir + '/view_0'
os.system('mkdir -p ' + des_dir)

r = global_scale * 3/4 + 3.
cam_height = np.sqrt(2)/2 * r
cam_dis = np.sqrt(2)/2 * r

camPos = np.array([0., cam_height, 0.])
camAngle = np.array([0., -np.deg2rad(90.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

for i in range(time_step):
    # pyflex.step(capture=1, path=os.path.join(des_dir, 'render_%d.tga' % i))
    pyflex.step()

obs = pyflex.render(render_depth=True).reshape(720, 720, 5)
print('obs.shape', obs.shape)

# save obs and camera_params
np.save(os.path.join(des_dir, 'obs.npy'), obs)

pyflex.clean()