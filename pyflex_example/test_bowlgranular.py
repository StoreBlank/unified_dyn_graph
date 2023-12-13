import os
import numpy as np
import pyflex
import time

from utils_env import rand_float, quatFromAxisAngle, degs_to_quat

pyflex.init(False)

radius = 0.02
bowl_pos = [-0.3, 0.5, -0.3]
bowl_mass = 1e100
bowl_scale = 1.5

num_granular_ft = [5, 50, 5] # low 5, medium 10, high 20
granular_scale = 0.15
pos_granular = [0., 1., 0.]
granular_dis = 0.

spoon_scale = 1.
spoon_mass = 10.
spoon_rotation = 0.1

draw_mesh = 1

scene_params = np.array([radius, *bowl_pos, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                                draw_mesh, bowl_mass, bowl_scale, spoon_scale, spoon_mass, spoon_rotation])

temp = np.array([0])
pyflex.set_scene(35, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)


# add table
table_height = 0.5
halfEdge = np.array([4., table_height, 4.])
center = np.array([0.0, 0.0, 0.0])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.ones(3) * (160. / 255.)
pyflex.add_box(halfEdge, center, quats, hideShape, color)
table_shape_states = np.concatenate([center, center, quats, quats])
# print('table_shape_states', table_shape_states.shape) # (14,)

obj_shape_states = np.zeros((2, 14))
bowl_scale = 15.
bowl_trans = np.array([0.5, table_height+0.6, 0.5])
bowl_quat = quatFromAxisAngle(np.array([1., 0., 0.]), np.deg2rad(270.))
bowl_color = np.array([204/255, 204/255, 1.])
pyflex.add_mesh('/home/baoyu/2023/unified_dyn_graph/assets/mesh/bowl.obj', bowl_scale, 0, 
                bowl_color, bowl_trans, bowl_quat, False)
obj_shape_states[0] = np.concatenate([bowl_trans, bowl_trans, bowl_quat, bowl_quat])

spoon_scale = 12.
spoon_trans = np.array([0.5, table_height+0.1, -1.])
spoon_quat_axis = np.array([1., 0., 0.])
spoon_quat = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(270.))
spoon_color = np.array([204/255, 204/255, 1.])
pyflex.add_mesh('/home/baoyu/2023/unified_dyn_graph/assets/mesh/spoon.obj', spoon_scale, 0,
                spoon_color, spoon_trans, spoon_quat, False)
# obj_shape_states[1] = np.concatenate([spoon_trans, spoon_trans, spoon_quat, spoon_quat])
spoon_pos_prev = spoon_trans
spoon_quat_prev = spoon_quat

# shape_states = np.zeros((3, 14))
# shape_states[0] = table_shape_states
# shape_states[1:] = obj_shape_states
# pyflex.set_shape_states(shape_states)

## Light setting
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

# camPos = np.array([0., 10, 0.])
# camAngle = np.array([0., -np.deg2rad(90.), 0.])
cam_dis = 4.
cam_height = 6.
camPos = np.array([cam_dis, cam_height, cam_dis])
camAngle = np.array([np.deg2rad(45.), -np.deg2rad(45.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

pyflex.step()

lim_y = 2.
lim_z = 0.6
lim_x = 0.5

# update the shape states for each time step
for i in range(2800):
    n_stay_still = 40
    n_up = 1500
    n_scoop = 2000
    
    if i < n_stay_still:
        angle_cur = 0.
        spoon_angle_delta = 0.
        spoon_pos_delta = np.zeros(3, dtype=np.float32)
    elif n_stay_still <= i < n_up:
        # spoon y position
        scale = 0.003
        spoon_pos_delta[1] = scale
        spoon_trans[1] += spoon_pos_delta[1]
        spoon_trans[1] = np.clip(spoon_trans[1], 0., lim_y)
        
        # spoon z position
        scale = 0.001
        spoon_pos_delta[2] = scale
        spoon_trans[2] += spoon_pos_delta[2]
        spoon_trans[2] = np.clip(spoon_trans[2], -2.0, lim_z)
        
        # spoon x position
        scale = 0.002
        spoon_pos_delta[0] = scale
        spoon_trans[0] -= spoon_pos_delta[0]
        spoon_trans[0] = np.clip(spoon_trans[0], -0.3, lim_x)
        
    elif n_up <= i < n_scoop:
        # spoon y position
        scale = 0.003
        spoon_pos_delta[1] = -scale
        spoon_trans[1] += spoon_pos_delta[1]
        spoon_trans[1] = np.clip(spoon_trans[1], table_height+0.6, lim_y)
        
        # spoon x position
        scale = 0.001
        spoon_pos_delta[0] = scale
        spoon_trans[0] += spoon_pos_delta[0]
        spoon_trans[0] = np.clip(spoon_trans[0], -0.3, 0.3)
        
        # spoon angle
        scale = 0.002
        # spoon_angle_delta[2] = scale
        spoon_quat_axis += np.array([0., 0., scale])
        spoon_quat_axis[2] = np.clip(spoon_quat_axis[2], 0., 0.5)
        spoon_quat = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(270.))
        
    elif n_scoop <= i:
        # spoon y position
        scale = 0.003
        spoon_pos_delta[1] = scale
        spoon_trans[1] += spoon_pos_delta[1]
        spoon_trans[1] = np.clip(spoon_trans[1], table_height+0.4, lim_y)
        
        # spoon angle
        scale = 0.001
        # spoon_angle_delta[2] = scale
        spoon_quat_axis -= np.array([0., 0., scale])
        spoon_quat_axis[2] = np.clip(spoon_quat_axis[2], 0.1, 0.4)
        spoon_quat = quatFromAxisAngle(spoon_quat_axis, np.deg2rad(270.))
        
    
    # set shape states
    shape_states = np.zeros((3, 14))
    shape_states[0] = table_shape_states
    
    # set shape state for table
    shape_states[0] = table_shape_states
    
    # set shape state for bowl
    shape_states[1] = obj_shape_states[0]
    
    # set shape state for spoon
    shape_states[2, :3] = spoon_trans
    shape_states[2, 3:6] = spoon_pos_prev
    shape_states[2, 6:10] = spoon_quat
    shape_states[2, 10:] = spoon_quat_prev
    
    spoon_pos_prev = spoon_trans
    spoon_quat_prev = spoon_quat
    
    pyflex.set_shape_states(shape_states)
    
    pyflex.step()
    

pyflex.clean()