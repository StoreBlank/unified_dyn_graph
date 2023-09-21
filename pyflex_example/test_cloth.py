import os
import cv2
import numpy as np
import pyflex
import time
import torch
import scipy.misc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=int, default=200)
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--screenWidth', type=int, default=720)
parser.add_argument('--screenHeight', type=int, default=720)
parser.add_argument('--fabric_type', type=int, default=0, help='0: Cloth, 1: shirt, 2: pants')
args = parser.parse_args()

np.random.seed(0)

screenWidth = args.screenWidth
screenHeight = args.screenHeight
time_step = args.time_step # 120
fabric_type = args.fabric_type

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def rotate_vector_2d(vector, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)

pyflex.init(False)

# offset
radius = 0.05
offset_x = -1.
offset_y = 0.06
offset_z = -1.

if fabric_type == 0:
    folder_dir = '../ptcl_data/cloth'
    os.system('mkdir -p ' + folder_dir)
    # parameters of the shape
    dimx = rand_int(25, 35)    # dimx, width
    dimy = rand_int(25, 35)    # dimy, height
    dimz = 0
    # the actuated points
    ctrl_idx = np.array([
        0, dimx // 2, dimx - 1,
        dimy // 2 * dimx,
        dimy // 2 * dimx + dimx - 1,
        (dimy - 1) * dimx,
        (dimy - 1) * dimx + dimx // 2,
        (dimy - 1) * dimx + dimx - 1])

    offset_x = -dimx * radius / 2.
    offset_y = 0.06
    offset_z = -dimy * radius / 2.

elif fabric_type == 1:
    folder_dir = '../ptcl_data/shirt'
    os.system('mkdir -p ' + folder_dir)
    # parameters of the shape
    dimx = rand_int(16, 25)     # width of the body
    dimy = rand_int(30, 35)     # height of the body
    dimz = 7                    # size of the sleeves
    # the actuated points
    ctrl_idx = np.array([
        dimx * dimy,
        dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4,
        dimx * dimy + (1 + dimz) * dimz // 2 + dimz * (dimz - 1),
        dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4 + \
            (1 + dimz) * dimz // 2 + dimz * dimz - 1,
        dimy // 2 * dimx,
        dimy // 2 * dimx + dimx - 1,
        (dimy - 1) * dimx,
        dimy * dimx - 1])

    offset_x = -(dimx + dimz * 4) * radius / 2.
    offset_y = 0.06
    offset_z = -dimy * radius / 2.

elif fabric_type == 2:
    folder_dir = '../ptcl_data/pant'
    os.system('mkdir -p ' + folder_dir)
    # parameters of the shape
    dimx = rand_int(9, 13) * 2 # width of the pants
    dimy = rand_int(6, 11)      # height of the top part
    dimz = rand_int(24, 31)     # height of the leg
    # the actuated points
    ctrl_idx = np.array([
        0, dimx - 1,
        (dimy - 1) * dimx,
        (dimy - 1) * dimx + dimx - 1,
        dimx * dimy + dimz // 2 * (dimx - 4) // 2,
        dimx * dimy + (dimz - 1) * (dimx - 4) // 2,
        dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
            dimz // 2 * (dimx - 4) // 2 + (dimx - 4) // 2 - 1,
        dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
            dimz * (dimx - 4) // 2 - 1])

    offset_x = -dimx * radius / 2.
    offset_y = 0.06
    offset_z = -(dimy + dimz) * radius / 2.


# physical param
stiffness = rand_float(0.4, 1.0)
stretchStiffness = stiffness
bendStiffness = stiffness
shearStiffness = stiffness

dynamicFriction = 0.6
staticFriction = 1.0
particleFriction = 0.6

invMass = 1.0

# other parameters
windStrength = 0.0
draw_mesh = 1.

# set up environment
scene_params = np.array([
    offset_x, offset_y, offset_z,
    fabric_type, dimx, dimy, dimz,
    ctrl_idx[0], ctrl_idx[1], ctrl_idx[2], ctrl_idx[3],
    ctrl_idx[4], ctrl_idx[5], ctrl_idx[6], ctrl_idx[7],
    stretchStiffness, bendStiffness, shearStiffness,
    dynamicFriction, staticFriction, particleFriction,
    invMass, windStrength, draw_mesh])

pyflex.set_scene(27, scene_params, 0)

## Light setting
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

r = 5.
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
print('camera_params', camera_intrinsic_params)
print('projMat: \n', projMat)

# camera extrinsic parameters
viewMat = pyflex.get_viewMatrix().reshape(4, 4).T
print('viewMat: \n', viewMat)

# let the cloth drop
for i in range(time_step):
    action = np.zeros(4)
    pyflex.step(action)

print('n_particles', pyflex.get_n_particles())

# render
obs = pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)
print('obs.shape', obs.shape)

# save obs and camera_params
np.save(os.path.join(des_dir, 'obs.npy'), obs)
np.save(os.path.join(des_dir, 'camera_intrinsic_params.npy'), camera_intrinsic_params)
np.save(os.path.join(des_dir, 'camera_extrinsic_matrix.npy'), viewMat)


pyflex.clean()

