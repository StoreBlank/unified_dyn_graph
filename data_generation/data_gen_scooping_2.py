import os
import numpy as np
import pyflex
import time
import cv2

from utils_env import rand_float, rand_int, quatFromAxisAngle, quaternion_multuply
from data_generation.utils import add_table, set_light, set_camera
from data_generation.utils import init_multiview_camera, get_camera_intrinsics, get_camera_extrinsics
from data_generation.utils import fps_with_idx, randomize_pos, render

camera_view = 4

def data_gen_scooping(info):
    epi_start_time = time.time()
    # info
    debug = info['debug']
    data_root_dir = info['data_root_dir']
    headless = info['headless']
    
    epi = info['epi']
    n_time_step = info['n_time_step']
    n_scoop = info['n_scoop']
    
    num_sample_points = info['num_sample_points']
    
    # create folder
    folder_dir = os.path.join(data_root_dir, 'granular_scooping')
    os.system('mkdir -p ' + folder_dir)
    
    epi_dir = os.path.join(folder_dir, "episode_%d" % epi)
    os.system("mkdir -p %s" % epi_dir)
    
    pyflex.init(headless)
    # np.random.seed(epi)
    ## set scene
    radius = 0.03
    
    num_granular_ft_x = 5
    num_granular_ft_y = 3
    num_granular_ft_z = 5
    num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z] 
    num_granular = int(num_granular_ft_x * num_granular_ft_y * num_granular_ft_z)
    
    granular_scale = 0.2
    pos_granular = [-0.5, 1., 0.]
    granular_dis = 0.

    draw_mesh = 0
    
    shapeCollisionMargin = 0.05
    collisionDistance = 0.03 #granular_scale * 0.1
    
    dynamic_friction = 0.3
    granular_mass = 0.1

    scene_params = np.array([radius, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                            draw_mesh, shapeCollisionMargin, collisionDistance, dynamic_friction, 
                            granular_mass])

    temp = np.array([0])
    pyflex.set_scene(35, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)

    ## set env
    ## add table
    table_height = 0.5
    table_side = 5.
    table_shape_states = add_table(table_height, table_side)
    
    ## add spatula
    spoon_scale = 0.4
    # spoon_pos_x = 4.0
    # spoon_pos_z = 0.5
    spoon_pos_y = table_height+0.98 #0.93-0.95
    # spoon_pos = np.array([spoon_pos_x, spoon_pos_y, spoon_pos_z])
    spoon_pos_origin = randomize_pos(spoon_pos_y)
    spoon_pos = spoon_pos_origin.copy()
    
    spoon_quat_axis_origin = np.array([0., 0., 1.])
    angle_origin = 30.
    spoon_quat_origin = quatFromAxisAngle(spoon_quat_axis_origin, np.deg2rad(angle_origin))
    spoon_quat_axis = np.array([0., 1., 0.])
    
    angle = np.random.randint(0, 360)
    spoon_quat_2 = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(angle))
    spoon_quat = quaternion_multuply(spoon_quat_2, spoon_quat_origin)
    
    spoon_color = np.array([204/255, 204/255, 1.])
    pyflex.add_mesh('assets/mesh/spatula.obj', spoon_scale, 0,
                    spoon_color, spoon_pos, spoon_quat, False)
    spoon_pos_prev = spoon_pos
    spoon_quat_prev = spoon_quat
    

    ## Light and camera setting
    screenHeight, screenWidth = 720, 720
    cam_dis, cam_height = 6., 10.
    set_light(screenHeight, screenWidth)
    set_camera(cam_dis, cam_height, camera_view)
    camPos_list, camAngle_list, cam_intrinsic_params, cam_extrinsic_matrix = init_multiview_camera(cam_dis, cam_height)
            
    pyflex.step()

    ## update the shape states for each time step
    count = 0
    step_list = []
    particle_pos_list = []
    tool_pos_list = []
    tool_quat_list = []
    
    for p in range(n_scoop):
        for i in range(n_time_step):
            angle = 30
            
            n_stay_still = 40
            n_scoop = 300
            n_up = 600
            
            n_move = 2000
            
            if n_stay_still < i < n_scoop:
                # change spoon x position
                spoon_pos[0] -= 0.01
                spoon_pos[0] = np.clip(spoon_pos[0], 2.0, spoon_pos_x) 
            if n_scoop < i < n_up:
                # change spoon y position
                spoon_pos[1] += 0.01
                spoon_pos[1] = np.clip(spoon_pos[1], spoon_pos_y, spoon_pos_y+1.5)
            if n_up < i:
                # change spoon z position
                # spoon_pos[2] -= 0.005
                # spoon_pos[2] = np.clip(spoon_pos[2], -0.5, spoon_pos_z)
                
                # change spoon angle
                spoon_quat_axis[2] += 0.001
                spoon_quat_axis[2] = np.clip(spoon_quat_axis[2], -2.0, 2.0)
                spoon_quat_axis[0] += 0.005
                spoon_quat_axis[0] = np.clip(spoon_quat_axis[0], -2.0, 10.0)
                spoon_quat = quatFromAxisAngle(spoon_quat_axis_origin, np.deg2rad(angle))
            
            # set shape states
            shape_states = np.zeros((2, 14))
            shape_states[0] = table_shape_states
            
            # set shape state for table
            shape_states[0] = table_shape_states
            
            # set shape state for spoon
            shape_states[1, :3] = spoon_pos
            shape_states[1, 3:6] = spoon_pos_prev
            shape_states[1, 6:10] = spoon_quat
            shape_states[1, 10:] = spoon_quat_prev
            
            spoon_pos_prev = spoon_pos
            spoon_quat_prev = spoon_quat
            
            pyflex.set_shape_states(shape_states)
            
            if not debug and i % 2 == 0:
                num_cam = len(camPos_list)
                for j in range(1):
                    pyflex.set_camPos(camPos_list[j])
                    pyflex.set_camAngle(camAngle_list[j])
                    
                    # create dir with cameras
                    cam_dir = os.path.join(folder_dir, 'camera_%d' % (j))
                    os.system('mkdir -p ' + cam_dir)
                    
                    # save camera params
                    
                    # save rgb images
                    img = render(screenHeight, screenWidth)
                    cv2.imwrite(os.path.join(cam_dir, '%d_color.jpg' % count), img[:, :, :3][..., ::-1])
            
                count += 1
            
            pyflex.step()

    pyflex.clean()

### data generation for scooping
info = {
    'epi': 0,
    'n_time_step': 1000,
    'n_scoop': 1,
    'num_sample_points': 2000,
    "headless": False,
    "data_root_dir": "data_dense",
    "debug": True,
}

data_gen_scooping(info)