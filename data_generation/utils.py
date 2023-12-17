import numpy as np
import pyflex

from utils_env import rand_float

def init_multiview_camera(cam_dis = 3, cam_height = 4.5):
    camPos_list = []
    camAngle_list = []

    rad_list = np.deg2rad(np.array([0., 90., 180., 270.]) + 45.)
    cam_x_list = np.array([cam_dis, cam_dis, -cam_dis, -cam_dis])
    cam_z_list = np.array([cam_dis, -cam_dis, -cam_dis, cam_dis])

    for i in range(len(rad_list)):
        camPos_list.append(np.array([cam_x_list[i], cam_height, cam_z_list[i]]))
        camAngle_list.append(np.array([rad_list[i], -np.deg2rad(45.), 0.]))
    
    cam_intrinsic_params = np.zeros([len(camPos_list), 4]) # [fx, fy, cx, cy]
    cam_extrinsic_matrix = np.zeros([len(camPos_list), 4, 4]) # [R, t]
    
    return camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix

def render(screenHeight, screenWidth, no_return=False):
    pyflex.step()
    if no_return:
        return
    else:
        return pyflex.render(render_depth=True).reshape(screenHeight, screenWidth, 5)

def randomize_pos(init_y):
    # initial start position
    x_range_min, x_range_max = 0., 4. # range for x if not in (2,3)
    z_range_min, z_range_max = 0., 4. # range for z if not in (2,3)
    range_min, range_max = 3., 4.
    # randomly decide whether x or z will be in the range
    if np.random.choice(['x', 'z']) == 'x':
        pos_x = rand_float(range_min, range_max) * np.random.choice([-1., 1.])
        pos_z = rand_float(z_range_min, z_range_max) * np.random.choice([-1., 1.])
    else:
        pos_x = rand_float(x_range_min, x_range_max) * np.random.choice([-1., 1.])
        pos_z = rand_float(range_min, range_max) * np.random.choice([-1., 1.])
    
    pos_y = init_y
    pos = np.array([pos_x, pos_y, pos_z])
    return pos