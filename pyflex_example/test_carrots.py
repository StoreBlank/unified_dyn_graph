import os
import numpy as np
import pyflex
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=int, default=500)
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--screenWidth', type=int, default=720)
parser.add_argument('--screenHeight', type=int, default=720)
args = parser.parse_args()

screenWidth = args.screenWidth
screenHeight = args.screenHeight
time_step = args.time_step # 120

folder_dir = '../ptcl_data/carrots'
os.system('mkdir -p ' + folder_dir)

pyflex.init(True)

# scene_params = np.array([])

global_scale = 24 # default 24

np.random.seed(0)
rand_scale = np.random.uniform(0.09, 0.12) * global_scale / 8.0
max_scale = rand_scale
min_scale = rand_scale
blob_r = np.random.uniform(0.7, 1.0)
x = - blob_r * global_scale / 8.0
y = 0.5
z = - blob_r * global_scale / 8.0
inter_space = 1.5 * max_scale
num_x = int(abs(x/1.5) / max_scale + 1) * 2
num_y = 10
num_z = int(abs(z/1.5) / max_scale + 1) * 2
x_off = global_scale * np.random.uniform(-1./24., 1./24.)
z_off = global_scale * np.random.uniform(-1./24., 1./24.)
x += x_off
z += z_off
num_carrots = (num_x * num_z - 1) * 3
add_singular = 0.0
add_sing_x = -1
add_sing_y = -1
add_sing_z = -1
add_noise = 0.0

staticFriction = 1.0
dynamicFriction = 0.9
draw_skin = 1.0
min_dist = 10.0
max_dist = 20.0

scene_params = np.array([max_scale,
                        min_scale,
                        x,
                        y,
                        z,
                        staticFriction,
                        dynamicFriction,
                        draw_skin,
                        num_carrots,
                        min_dist,
                        max_dist,
                        num_x,
                        num_y,
                        num_z,
                        inter_space,
                        add_singular,
                        add_sing_x,
                        add_sing_y,
                        add_sing_z,
                        add_noise,])

temp = np.array([0])
pyflex.set_scene(22, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

## Light setting
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)


r = global_scale * 3/4
cam_height = np.sqrt(2)/2 * r
cam_dis = np.sqrt(2)/2 * r

## Camera setting
if args.view == 0: # top view
    des_dir = folder_dir + '/view_0'
    os.system('mkdir -p ' + des_dir)
    
    camPos = np.array([0., cam_height, 0.])
    camAngle = np.array([0., -np.deg2rad(90.), 0.])

elif args.view == 1: # positioned on positive x-axis, looking at the origin
    des_dir = folder_dir + '/view_1'
    os.system('mkdir -p ' + des_dir)
    
    camPos = np.array([cam_dis, cam_height, 0.])
    camAngle = np.array([np.deg2rad(90.), -np.deg2rad(45.), 0.])
    
elif args.view == 2: # positioned on positive z-axis, looking at the origin
    des_dir = folder_dir + '/view_2'
    os.system('mkdir -p ' + des_dir)
    
    camPos = np.array([0., cam_height, cam_dis])
    camAngle = np.array([np.deg2rad(0.), -np.deg2rad(45.), 0.])
    
elif args.view == 3: # positioned on negative x-axis, looking at the origin
    des_dir = folder_dir + '/view_3'
    os.system('mkdir -p ' + des_dir)
    
    camPos = np.array([-cam_dis, cam_height, 0.])
    camAngle = np.array([np.deg2rad(270.), -np.deg2rad(45.), 0.])
    
elif args.view == 4: # positioned on negative z-axis, looking at the origin
    des_dir = folder_dir + '/view_4'
    os.system('mkdir -p ' + des_dir)

    camPos = np.array([0., cam_height, -cam_dis])
    camAngle = np.array([np.deg2rad(180.), -np.deg2rad(45.), 0.])

print('camPos', camPos)
print('camAngle', camAngle)
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
# by viewMat @ point, transforms the point in world coordinates to the point in camera coordinates
# the camera coordinates follow OpenGL convention, +X right, +Y up, +Z points to the camera (look-at direction)
# In contrast, OpenCV and Open3D convention is +X right, +Y down, +Z points away from the camera
viewMat = pyflex.get_viewMatrix().reshape(4, 4).T
print('viewMat: \n', viewMat)

for i in range(time_step):
    pyflex.step()

# render
# The render function gives positive depth values for points in front of the camera
# In other words, it actually uses the Open3D version of the viewMat for projection to get the depth values
obs = pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)
print('obs.shape', obs.shape)

# save obs and camera_params
np.save(os.path.join(des_dir, 'obs.npy'), obs)
np.save(os.path.join(des_dir, 'camera_intrinsic_params.npy'), camera_intrinsic_params)
np.save(os.path.join(des_dir, 'camera_extrinsic_matrix.npy'), viewMat)

pyflex.clean()
