import os
import numpy as np
import pyflex
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=int, default=300)
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--screenWidth', type=int, default=720)
parser.add_argument('--screenHeight', type=int, default=720)
args = parser.parse_args()

np.random.seed(0)

screenWidth = args.screenWidth
screenHeight = args.screenHeight
time_step = args.time_step # 120
folder_dir = '../ptcl_data/mustar_bottle'
os.system('mkdir -p ' + folder_dir)

pyflex.init(False)

x = -0.5
y = 1.
z = 0.
size = 1.

scene_params = np.array([x, y, z, size])

pyflex.set_scene(25, scene_params, 0) 

## Light setting
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 2.0, 0.1]))
pyflex.set_light_fov(70.)

## Camera setting
if args.view == 0: # top view
    des_dir = folder_dir + '/view_0'
    os.system('mkdir -p ' + des_dir)
    
    cam_idx = 0
    cam_height = 5.
    rad = np.deg2rad(cam_idx)
    cam_dis = 0
    
    camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
    camAngle = np.array([rad, -np.deg2rad(90.), 0.])
    
elif args.view == 1: # lower right corner
    des_dir = folder_dir + '/view_1'
    os.system('mkdir -p ' + des_dir)
    cam_height = 5.
    camPos = np.array([cam_height/4, cam_height, cam_height/4])
    camAngle = np.array([np.deg2rad(45.), -np.deg2rad(70.), np.deg2rad(45.)])
    
elif args.view == 2: # upper right corner
    des_dir = folder_dir + '/view_2'
    os.system('mkdir -p ' + des_dir)
    cam_height = 5.
    camPos = np.array([cam_height/4, cam_height, -cam_height/4])
    camAngle = np.array([np.deg2rad(130.), -np.deg2rad(70.), np.deg2rad(45.)])
    
elif args.view == 3: # upper left corner
    des_dir = folder_dir + '/view_3'
    os.system('mkdir -p ' + des_dir)
    cam_height = 5.
    camPos = np.array([-cam_height/4, cam_height, -cam_height/4])
    camAngle = np.array([-np.deg2rad(130.), -np.deg2rad(70.), np.deg2rad(45.)])
    
elif args.view == 4: # lower left corner
    des_dir = folder_dir + '/view_4'
    os.system('mkdir -p ' + des_dir)
    cam_height = 5.
    camPos = np.array([-cam_height/4, cam_height, cam_height/4])
    camAngle = np.array([-np.deg2rad(45.), -np.deg2rad(70.), np.deg2rad(45.)])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

# camera intrinsic parameters
projMat = pyflex.get_projMatrix().reshape(4, 4).T 
cx = screenWidth / 2.0
cy = screenHeight / 2.0
fx = projMat[0, 0] * cx
fy = projMat[1, 1] * cy
camera_intrinsic_params = np.array([fx, fy, cx, cy])
print('camera_params', camera_intrinsic_params)
print('projMat: \n', projMat)

# camera extrinsic parameters
viewMat = pyflex.get_viewMatrix().reshape(4, 4).T
print('viewMat: \n', viewMat)

for i in range(time_step):
    pyflex.step()

# render
obs = pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)
print('obs.shape', obs.shape)

# save obs and camera_params
np.save(os.path.join(des_dir, 'obs.npy'), obs)
np.save(os.path.join(des_dir, 'camera_intrinsic_params.npy'), camera_intrinsic_params)
np.save(os.path.join(des_dir, 'camera_extrinsic_matrix.npy'), viewMat)

pyflex.clean()
