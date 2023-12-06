import os
import numpy as np
import pyflex
import time

from utils_env import rand_float, quatFromAxisAngle, degs_to_quat

pyflex.init(False)

radius = 0.2
# water pos
lower_x = 0.
lower_y = 2.
lower_z = 0.
# water cube size
dim_x = 10
dim_y = 5
dim_z = 10

lower = np.array([lower_x, lower_y, lower_z])
fluid_pos = np.array([dim_x, dim_y, dim_z])

viscosity = 200 #TODO
cohesion = 0.05 #TODO
shapeCollisionMargin = 1e-100
draw_mesh = 1

dynamic_friction = 0.9
scene_params = np.array([radius, *lower, *fluid_pos, draw_mesh, 
                         viscosity, cohesion, shapeCollisionMargin, dynamic_friction])

temp = np.array([0])
pyflex.set_scene(36, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)
pyflex.set_fluid_color(np.array([153/255, 76/255, 0/255, 0.0]))

num_particles = pyflex.get_n_particles()
print('num_particles', num_particles)

## add box
table_height = 0.5
table_length = 6.
halfEdge = np.array([table_length, table_height, table_length])
center = np.array([0.0, 0.0, 0.0])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.ones(3) * (160. / 255.)
pyflex.add_box(halfEdge, center, quats, hideShape, color)
table_shape_states = np.concatenate([center, center, quats, quats])
# print('table_shape_states', table_shape_states.shape) # (14,)

obj_shape_states = np.zeros((2, 14))
## add bread
bread_scale = 0.3
bread_pos = np.array([0.5, table_height+0.1, 0.5])
bread_quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.deg2rad(0.))
bread_color = np.array([255/255, 153/255, 51/255])
pyflex.add_mesh('/home/baoyu/2023/unified_dyn_graph/assets/mesh/bread.obj', bread_scale, 0,
                bread_color, bread_pos, bread_quat, False)
obj_shape_states[0] = np.concatenate([bread_pos, bread_pos, bread_quat, bread_quat])

## add knife
knife_scale = 0.03

knife_init_x = 2.5
knife_init_y = 1.5
knife_init_z = 1.0
knife_pos = np.array([knife_init_x, knife_init_y, knife_init_z])

deg_xyz = np.array([0., -90., 0.])
quaternion = degs_to_quat(deg_xyz)
knife_quat = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])

knife_color = np.array([128/255, 128/255, 128/255])
pyflex.add_mesh('/home/baoyu/2023/unified_dyn_graph/assets/mesh/butter_knife.obj', knife_scale, 0,
                knife_color, knife_pos, knife_quat, False)
knife_pos_prev = knife_pos
knife_quat_prev = knife_quat

## Light setting
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

# camPos = np.array([0., 10, 0.])
# camAngle = np.array([0., -np.deg2rad(90.), 0.])
cam_dis = 5.
cam_height = 8.
camPos = np.array([cam_dis, cam_height, cam_dis])
camAngle = np.array([np.deg2rad(45.), -np.deg2rad(45.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)

pyflex.step()




# update the shape states for each time step
for i in range(4000):
    n_stay_still = 40.
    n_first_move = 200 #400
    n_first_slice = 1100 #900
    n_first_up = 1300 #200
    
    n_second_move = 1400
    n_second_slice = 2300
    n_second_up = 2500
    
    n_third_move = 2800
    n_third_slice = 3700
    n_third_move_up = 3900
    
    
    
    ## first slice
    angle = 40
    x_to_jam = 0.8
    y_to_jam = table_height+0.5
    if n_stay_still < i < n_first_move:
        # change knife x position
        knife_pos[0] -= 0.01
        knife_pos[0] = np.clip(knife_pos[0], x_to_jam, 2.5)
        # chang knife y position
        knife_pos[1] += 0.01
        knife_pos[1] = np.clip(knife_pos[1], y_to_jam, 1.5)
        # change knife angle
        knife_quat_axis = np.array([0.0, 0., 1.0])
        knife_quat = quatFromAxisAngle(knife_quat_axis, np.deg2rad(angle))
    if n_first_move < i < n_first_slice:
        # chang knife y position
       knife_pos[1] -= 0.005
       knife_pos[1] = np.clip(knife_pos[1], y_to_jam, 1.5)
       # change knife x position
       knife_pos[0] += 0.001
       knife_pos[0] = np.clip(knife_pos[0], x_to_jam, 2.0)
    if n_first_slice < i < n_first_up:
        # change knife y position
        knife_pos[1] += 0.01
        knife_pos[1] = np.clip(knife_pos[1], y_to_jam, 1.5)
        
    ## second slice
    if n_first_slice < i < n_second_move:
        x_to_jam = 0.
        # change knife angle
        knife_quat_axis = np.array([0.0, 0., 1.0])
        angle = 135
        knife_quat = quatFromAxisAngle(knife_quat_axis, np.deg2rad(angle))
        # change knife x position
        knife_pos[0] -= 0.01
        knife_pos[0] = np.clip(knife_pos[0], x_to_jam, 2.0)
    if n_second_move < i < n_second_slice:
        # chang knife y position
       knife_pos[1] -= 0.005
       knife_pos[1] = np.clip(knife_pos[1], y_to_jam, 1.5)
       # change knife x position
       knife_pos[0] -= 0.001
       knife_pos[0] = np.clip(knife_pos[0], -3.0, 2.0)
    if n_second_slice < i < n_second_up:
        # change knife y position
        knife_pos[1] += 0.01
        knife_pos[1] = np.clip(knife_pos[1], y_to_jam, 1.5)
    
    ## third slice
    if n_second_up < i < n_third_move:
        z_to_jam = 2.5
        x_to_jam = 1.5
        # change knife angle
        deg_xyz = np.array([0., 0., 45.])
        quaternion = degs_to_quat(deg_xyz)
        knife_quat = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        # change knife x position
        knife_pos[0] += 0.01
        knife_pos[0] = np.clip(knife_pos[0], -3.0, x_to_jam)
        # change knife z position
        knife_pos[2] -= 0.01
        knife_pos[2] = np.clip(knife_pos[2], 0.5, z_to_jam)
    if n_third_move < i < n_third_slice:
        # chang knife y position
        knife_pos[1] -= 0.005
        knife_pos[1] = np.clip(knife_pos[1], y_to_jam, 1.5)
        # change knife z position
        knife_pos[2] += 0.001
        knife_pos[2] = np.clip(knife_pos[2], 0.5, z_to_jam)
    if n_third_slice < i < n_third_move_up:
        # change knife y position
        knife_pos[1] += 0.01
        knife_pos[1] = np.clip(knife_pos[1], y_to_jam, 1.5)
    
       
      
        
    
    # set shape states
    shape_states = np.zeros((3, 14))
    shape_states[0] = table_shape_states
    
    # set shape state for table
    shape_states[0] = table_shape_states
    
    # set shape state for bowl
    shape_states[1] = obj_shape_states[0]
    
    # set shape state for knife
    shape_states[2, :3] = knife_pos
    shape_states[2, 3:6] = knife_pos_prev
    shape_states[2, 6:10] = knife_quat
    shape_states[2, 10:] = knife_quat_prev
    
    knife_pos_prev = knife_pos
    knife_quat_prev = knife_quat
    
    pyflex.set_shape_states(shape_states)
    
    pyflex.step()
    

pyflex.clean()