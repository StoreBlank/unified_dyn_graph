import os
import numpy as np
import pyflex
import time


time_step = 500 # 120
# des_dir = 'test_FluidFall'
# os.system('mkdir -p ' + des_dir)

pyflex.init(False)

global_scale = 12

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

temp = np.array([0])
pyflex.set_scene(20, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

## Light setting
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

folder_dir = '../ptcl_data/coffee'
os.system('mkdir -p ' + folder_dir)

des_dir = folder_dir + '/view_0'
os.system('mkdir -p ' + des_dir)

r = global_scale * 3/4 + 5.
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