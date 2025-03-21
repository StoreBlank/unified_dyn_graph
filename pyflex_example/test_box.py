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
def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo

screenWidth = args.screenWidth
screenHeight = args.screenHeight
time_step = args.time_step # 120
folder_dir = '../ptcl_data/apples'
os.system('mkdir -p ' + folder_dir)

pyflex.init(False)

global_scale = 1

n_instance = 3
dynamic_friction = 0.1
gravity = -9.8
restitution = 0.1
low_bound = 0.09
draw_mesh = True
# scale = 0.2 * global_scale / 8.0
scale = 1

scene_params = np.zeros(n_instance * 3 + 4)
scene_params[0] = n_instance
scene_params[1] = gravity
scene_params[2] = scale

for i in range(n_instance):
    x = rand_float(0, 0.1)
    y = rand_float(low_bound, low_bound + 0.01)
    z = rand_float(0, 0.1)
    print('x, y, z', x, y, z)
    
    scene_params[3*i + 3] = x
    scene_params[3*i + 4] = y
    scene_params[3*i + 5] = z
    
    low_bound += 0.21

if draw_mesh:
    scene_params[n_instance*3 + 3] = 1 # draw mesh

# pyflex.set_scene(3, scene_params, 0) # 3: RigidFall
pyflex.set_scene(24, scene_params, 0) # 24, 25

## Light setting
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

r = 20.
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