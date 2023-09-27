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
parser.add_argument('--type', type=int, default=6)
args = parser.parse_args()

np.random.seed(0)

screenWidth = args.screenWidth
screenHeight = args.screenHeight
time_step = args.time_step # 120

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
y = 1.
z = -0.5
size = 1.
obj_type = args.type

# if obj_type == 3:
#     folder_dir = '../ptcl_data/cracker_box'
# elif obj_typee == 4:
#     folder_dir = '../ptcl_data/sugar_box'
# elif obj_type == 5:
#     folder_dir = '../ptcl_data/tomato_soup_can'
# elif obj_type == 6:
#     folder_dir = '../ptcl_data/mustard_bottle'
# elif obj_type == 7:
#     folder_dir = '../ptcl_data/tuna_fish_can'
# elif obj_type == 8:
#     folder_dir = '../ptcl_data/pudding_box'
# elif obj_type == 9:
#     folder_dir = '../ptcl_data/gelatin_box'
# elif obj_type == 10:
#     folder_dir = '../ptcl_data/potted_meat_can'
# elif obj_type == 12:
#     folder_dir = '../ptcl_data/strawberry'
# elif obj_type == 13:
#     folder_dir = '../ptcl_data/apple'
# elif obj_type == 14:
#     folder_dir = '../ptcl_data/lemon'
# elif obj_type == 15:
#     folder_dir = '../ptcl_data/peach'
# elif obj_type == 16:
#     folder_dir = '../ptcl_data/pear'
# elif obj_type == 17:
#     folder_dir = '../ptcl_data/orange'
# elif obj_type == 19:
#     folder_dir = '../ptcl_data/pitcher_base'
# elif obj_type == 21:
#     folder_dir = '../ptcl_data/bleach_cleanser'
# elif obj_type == 24:
#     folder_dir = '../ptcl_data/bowl'
# elif obj_type == 25:
#     folder_dir = '../ptcl_data/mug'
# elif obj_type == 35:
#     folder_dir = '../ptcl_data/power_drill'
# elif obj_type == 36:
#     folder_dir = '../ptcl_data/wood_block'
# elif obj_type == 37:
#     folder_dir = '../ptcl_data/scissors'

# folder_dir = '../ptcl_data/single_ycb' 
folder_dir = '../ptcl_data/mustard_bottle'
os.system('mkdir -p ' + folder_dir)

scene_params = np.array([x, y, z, size, obj_type])

temp = np.array([0])
pyflex.set_scene(25, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 

print('n_particles', pyflex.get_n_particles())

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
