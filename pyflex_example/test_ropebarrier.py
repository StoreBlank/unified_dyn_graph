import os
import numpy as np
import pyflex
import time
from scipy.spatial.transform import Rotation

from utils_env import rand_float, quatFromAxisAngle, degs_to_quat
from utils_env import find_min_distance

camera_view = 1

def _set_pos(picker_pos, particle_pos):
    """For gripper and grasp task."""
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = shape_states[:, :3] #picker_pos
    shape_states[:, :3] = picker_pos
    pyflex.set_shape_states(shape_states)
    pyflex.set_positions(particle_pos)

pyflex.init(False)

### set scene
radius = 0.04 #0.03

length = 3.0 #rand_float(0.5, 2.5)
thickness = 3.0 #rand_float(1., 2.5)
scale = np.array([length, thickness, thickness]) * 50 # length, extension, thickness
cluster_spacing = 5. #rand_float(2, 8) # change the stiffness of the rope
dynamicFriction = 0.3 #rand_float(0.1, 0.7)

trans = [0., 0.5, 3.]

z_rotation = 0. #rand_float(70, 80)
y_rotation = 90. #np.random.choice([0, 30, 45, 60])
rot = Rotation.from_euler('xyz', [0, y_rotation, z_rotation], degrees=True)
rotate = rot.as_quat()

cluster_radius = 0.
cluster_stiffness = 0.2

link_radius = 0. 
link_stiffness = 1.

global_stiffness = 0.

surface_sampling = 0.
volume_sampling = 4.

skinning_falloff = 5.
skinning_max_dist = 100.

cluster_plastic_threshold = 0.
cluster_plastic_creep = 0.

particleFriction = 0.25

draw_mesh = 0

relaxtion_factor = 1.
collisionDistance = radius * 0.5

scene_params = np.array([*scale, *trans, radius, 
                                cluster_spacing, cluster_radius, cluster_stiffness,
                                link_radius, link_stiffness, global_stiffness,
                                surface_sampling, volume_sampling, skinning_falloff, skinning_max_dist,
                                cluster_plastic_threshold, cluster_plastic_creep,
                                dynamicFriction, particleFriction, draw_mesh, relaxtion_factor, 
                                *rotate, collisionDistance])

temp = np.array([0])
pyflex.set_scene(26, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0) 


### rope statistics
particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
new_particle_pos = particle_pos.copy()
particle_inv_mass = particle_pos[:, 3]

num_particles = particle_pos.shape[0]
print('num_particles', num_particles)

particle_pos_x, particle_pos_y, particle_pos_z = particle_pos[:, 0], particle_pos[:, 1], particle_pos[:, 2]
particle_pos_x_min, particle_pos_x_max = np.min(particle_pos_x), np.max(particle_pos_x)
particle_pos_y_min, particle_pos_y_max = np.min(particle_pos_y), np.max(particle_pos_y)
particle_pos_z_min, particle_pos_z_max = np.min(particle_pos_z), np.max(particle_pos_z)
print('particle_pos_x_min', particle_pos_x_min, ';particle_pos_x_max', particle_pos_x_max)
print('particle_pos_y_min', particle_pos_y_min, ';particle_pos_y_max', particle_pos_y_max)
print('particle_pos_z_min', particle_pos_z_min, ';particle_pos_z_max', particle_pos_z_max)
rope_len_x, rope_len_y, rope_len_z = particle_pos_x_max - particle_pos_x_min, particle_pos_y_max - particle_pos_y_min, particle_pos_z_max - particle_pos_z_min
print('rope_len_x', rope_len_x, ';rope_len_y', rope_len_y, ';rope_len_z', rope_len_z)

## Light setting
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

## camera setting
cam_dis = 6.
cam_height = 8.
if camera_view == 1:
    camPos = np.array([cam_dis, cam_height, cam_dis])
    camAngle = np.array([np.deg2rad(45.), -np.deg2rad(45.), 0.])
elif camera_view == 2:
    camPos = np.array([cam_dis, cam_height, -cam_dis])
    camAngle = np.array([np.deg2rad(45.+90.), -np.deg2rad(45.), 0.])
elif camera_view == 3:
    camPos = np.array([-cam_dis, cam_height, -cam_dis])
    camAngle = np.array([np.deg2rad(45.+180.), -np.deg2rad(45.), 0.])
elif camera_view == 4:
    camPos = np.array([-cam_dis, cam_height, cam_dis])
    camAngle = np.array([np.deg2rad(45.+270.), -np.deg2rad(45.), 0.])

pyflex.set_camPos(camPos)
pyflex.set_camAngle(camAngle)


### set env
## add table
table_height = 0.5
halfEdge = np.array([4., table_height, 4.])
center = np.array([0.0, 0.0, 0.0])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.ones(3) * (160. / 255.)
pyflex.add_box(halfEdge, center, quats, hideShape, color)
table_shape_states = np.concatenate([center, center, quats, quats])

box_shape_states = np.zeros((3, 14))
## add box for the barrier
np.random.seed(0)
# box 1
box_side = 0.1
halfEdge = np.ones(3) * box_side
box_x_1, box_z_1 = -1, -1
box_1_center = np.array([box_x_1, table_height+box_side, box_z_1])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.array([1., 0., 0.]) 
pyflex.add_box(halfEdge, box_1_center, quats, hideShape, color)
box_shape_states[0] = np.concatenate([box_1_center, box_1_center, quats, quats])
# box 2
box_side = 0.1
halfEdge = np.ones(3) * box_side
box_x_2, box_z_2 = 1, 1
box_2_center = np.array([box_x_2, table_height+box_side, box_z_2])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.array([1., 0., 0.]) 
pyflex.add_box(halfEdge, box_2_center, quats, hideShape, color)
box_shape_states[1] = np.concatenate([box_2_center, box_2_center, quats, quats])

# find the index of min z
min_z_idx, max_z_idx = np.argmin(particle_pos_z), np.argmax(particle_pos_z)
# fixed one side of the rope
box_side = 0.2
halfEdge = np.ones(3) * box_side
center = np.array([np.median(particle_pos_x), table_height+box_side, particle_pos_z_max])
quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
hideShape = 0
color = np.ones(3) * (160. / 255.)
pyflex.add_box(halfEdge, center, quats, hideShape, color)
box_shape_states[2] = np.concatenate([center, center, quats, quats])

# random pick a point on the rope
np.random.seed(0)
pick_point_idx = np.random.randint(0, num_particles)
new_particle_pos[pick_point_idx, 1] = particle_pos[pick_point_idx, 1] + 2.
new_particle_pos[pick_point_idx, 3] = 0.
pyflex.set_positions(new_particle_pos)

pyflex.step()

shape_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
num_shape_pos = shape_pos.shape[0]
# print('num_shape_pos', num_shape_pos)
# print('shape_pos', shape_pos[:, :3])

box1_min_dist, box1_pick_index = find_min_distance(box_1_center, particle_pos[:, :3], 10)
box2_min_dist, box2_pick_index = find_min_distance(box_2_center, particle_pos[:, :3], 10)
# print('box1_pick_index', box1_pick_index, ';box2_pick_index', box2_pick_index)

# end_point_1 = box_1_center + np.array([box_side - 0.5, 2., 0.])
# end_point_2 = box_2_center + np.array([box_side + 0.5, 2., 0.])

end_point_1 = new_particle_pos[box1_pick_index[0], :3]
pos_prev = particle_pos.copy()
for i in range(2000):
    
    # find the cloest point to the box 1
    
    # print('end_point_1', end_point_1)
    # for index in box1_pick_index:
    #     new_particle_pos[index, 1] += 0.01 
    #     new_particle_pos[index, 1] = np.clip(new_particle_pos[index, 1], 0.5, 2.)
    #     new_particle_pos[index, 3] = 0.
    
    end_point_1[1] += 0.001
    
    new_particle_pos[box1_pick_index, :3] = end_point_1
    new_particle_pos[box1_pick_index, 3] = 0.
    
    _set_pos(end_point_1, new_particle_pos)
        
    # set shape states
    shape_states = np.zeros((num_shape_pos, 14))
    shape_states[0] = table_shape_states
    for i in range(num_shape_pos-1):
        shape_states[i+1] = box_shape_states[i]
    
    pyflex.set_shape_states(shape_states)
    
    pyflex.step()
    

pyflex.clean()