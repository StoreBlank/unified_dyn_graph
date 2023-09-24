import os
import numpy as np
import pyflex
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=int, default=600)
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--screenWidth', type=int, default=720)
parser.add_argument('--screenHeight', type=int, default=720)
args = parser.parse_args()

np.random.seed(0)

screenWidth = args.screenWidth
screenHeight = args.screenHeight
time_step = args.time_step 

pyflex.init(False)

"""
ycb types:
3: cracker box
4: sugar box
5: tomato soup can
6: mustard bottle
7: potted meat can
8: pudding box
9: gelatin box
10: potted meat can: debug
12: strawberry
13: apple
14: lemon
15: peach
16: pear
17: orange
19: pitcher base
21: bleach cleanser
24: bowl
25: mug: debug
35: power drill
36: wood block
37: scissors: debug
"""

x = -0.5 # -0/5
y = 0.
z = -0.5
size = 1.

folder_dir = '../ptcl_data/multi_ycb' 
os.system('mkdir -p ' + folder_dir)

scene_params = np.array([x, y, z, size])

pyflex.set_scene(28, scene_params, 0) 

print('n_particles', pyflex.get_n_particles())

## Light setting
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

r = 8.
## Camera setting
if args.view == 0: # top view
    des_dir = folder_dir + '/view_0'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = np.sqrt(2)/2 * r
    
    camPos = np.array([0., cam_height, 0.])
    camAngle = np.array([0., -np.deg2rad(90.), 0.])
    
elif args.view == 1: # lower right corner
    des_dir = folder_dir + '/view_1'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = np.sqrt(2)/2 * r
    
    camPos = np.array([cam_dis, cam_height, 0.])
    # camAngle = np.array([np.deg2rad(45.), -np.deg2rad(70.), np.deg2rad(45.)])
    camAngle = np.array([np.deg2rad(90.), -np.deg2rad(45.), 0.])
    
elif args.view == 2: # upper right corner
    des_dir = folder_dir + '/view_2'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = np.sqrt(2)/2 * r
    
    camPos = np.array([0., cam_height, cam_dis])
    # camAngle = np.array([np.deg2rad(130.), -np.deg2rad(70.), np.deg2rad(45.)])
    camAngle = np.array([0., -np.deg2rad(45.), 0.])
    
elif args.view == 3: # upper left corner
    des_dir = folder_dir + '/view_3'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = -np.sqrt(2)/2 * r
    
    camPos = np.array([cam_dis, cam_height, 0.])
    # camAngle = np.array([-np.deg2rad(130.), -np.deg2rad(70.), np.deg2rad(45.)])
    camAngle = np.array([np.deg2rad(270.), -np.deg2rad(45.), 0.])
    
elif args.view == 4: # lower left corner
    des_dir = folder_dir + '/view_4'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = -np.sqrt(2)/2 * r
    
    camPos = np.array([0., cam_height, cam_dis])
    # camAngle = np.array([-np.deg2rad(45.), -np.deg2rad(70.), np.deg2rad(45.)])
    camAngle = np.array([np.deg2rad(180.), -np.deg2rad(45.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

# camera intrinsic parameters
projMat = pyflex.get_projMatrix().reshape(4, 4).T 
cx = screenWidth / 2.0
cy = screenHeight / 2.0
fx = projMat[0, 0] * cx
fy = projMat[1, 1] * cy
camera_intrinsic_params = np.array([fx, fy, cx, cy])
# print('camera_params', camera_intrinsic_params)
# print('projMat: \n', projMat)

# camera extrinsic parameters
viewMat = pyflex.get_viewMatrix().reshape(4, 4).T
# print('viewMat: \n', viewMat)

for i in range(time_step):
    pyflex.step()

# render
obs = pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)
# print('obs.shape', obs.shape)

# save obs and camera_params
np.save(os.path.join(des_dir, 'obs.npy'), obs)
np.save(os.path.join(des_dir, 'camera_intrinsic_params.npy'), camera_intrinsic_params)
np.save(os.path.join(des_dir, 'camera_extrinsic_matrix.npy'), viewMat)

pyflex.clean()
