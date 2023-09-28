import os
import numpy as np
import pyflex
import trimesh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=int, default=300)
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--screenWidth', type=int, default=720)
parser.add_argument('--screenHeight', type=int, default=720)
parser.add_argument('--type', type=int, default=6)
args = parser.parse_args()

screenWidth = args.screenWidth
screenHeight = args.screenHeight
time_step = args.time_step # 120

def load_cloth(path):
    """Load .obj of cloth mesh. Only quad-mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    This function was written by Zhenjia Xu
    email: xuzhenjia [at] cs (dot) columbia (dot) edu
    website: https://www.zhenjiaxu.com/
    """
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)
                             for n in line.replace('v ', '').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            assert(len(face) == 4)
            faces.append(face)

    triangle_faces = []
    for face in faces:
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(
                    sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)

    return np.array(vertices), np.array(triangle_faces),\
        np.array(list(stretch_edges)), np.array(
            list(bend_edges)), np.array(list(shear_edges))

# convert mesh to vertices
path = "/home/baoyu/2023/unified_dyn_graph/cloth3d/Tshirt2.obj"
retval = load_cloth(path)
mesh_verts = retval[0]
mesh_faces = retval[1]
mesh_stretch_edges, mesh_bend_edges, mesh_shear_edges = retval[2:]

num_particle = mesh_verts.shape[0]//3
# flattened_area = trimesh.load(path).area/2

## scene parameters
# cloth_pos [X, Y, Z] 0, 1, 2
# cloth_pos = [0, 0, 0] # tshirt
cloth_pos = [-0.6, 0, -0.7] # cloth

# cloth_size [dimx, dimz] 3, 4 # have not been applied to shirt 
cloth_size = [100, 100] # [100, 100] + r=0.01 one grid

# stiffness [stretch, bend, shear] 5, 6, 7
# mass 8; particle_r 9
np.random.seed(0)
stiffness = np.random.uniform(0.85, 0.95, 3)
cloth_mass = np.random.uniform(0.2, 2.0)
particle_r = 0.01 # default 0.00625

# render mode 10; flip_mesh 11
render_mode = 1 # 0: points, 1: mesh, 2: mesh + points
flip_mesh = 0

pyflex.init(False)

scene_params = np.array([
    *cloth_pos,
    *cloth_size, 
    *stiffness,
    cloth_mass,
    particle_r,
    render_mode,
    flip_mesh])

# Tshirt
# pyflex.set_scene(
#     29,
#     scene_params,
#     mesh_verts.reshape(-1),
#     mesh_stretch_edges.reshape(-1),
#     mesh_bend_edges.reshape(-1),
#     mesh_shear_edges.reshape(-1),
#     mesh_faces.reshape(-1),
#     0)

# Simple Cloth
temp = np.array([0])
pyflex.set_scene(29, scene_params, temp.astype(np.float64), temp, temp, temp, temp, 0)

print('n_particles', pyflex.get_n_particles())

## Light setting
pyflex.set_screenWidth(screenWidth)
pyflex.set_screenHeight(screenHeight)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.)

folder_dir = '../ptcl_data/shirt'
os.system('mkdir -p ' + folder_dir)

r = 5.
move_x = 0 
move_z = 0 
## Camera setting
if args.view == 0: # top view
    des_dir = folder_dir + '/view_0'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = np.sqrt(2)/2 * r
    
    camPos = np.array([0.+move_x, cam_height, 0.+move_z])
    camAngle = np.array([0., -np.deg2rad(90.), 0.])
    
elif args.view == 1: # lower right corner
    des_dir = folder_dir + '/view_1'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = np.sqrt(2)/2 * r
    
    camPos = np.array([cam_dis+move_x, cam_height, 0.+move_z])
    camAngle = np.array([np.deg2rad(90.), -np.deg2rad(45.), 0.])
    
elif args.view == 2: # upper right corner
    des_dir = folder_dir + '/view_2'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = np.sqrt(2)/2 * r
    
    camPos = np.array([0.+move_x, cam_height, cam_dis+move_z])
    camAngle = np.array([0., -np.deg2rad(45.), 0.])
    
elif args.view == 3: # upper left corner
    des_dir = folder_dir + '/view_3'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = -np.sqrt(2)/2 * r
    
    camPos = np.array([cam_dis+move_x, cam_height, 0.+move_z])
    camAngle = np.array([np.deg2rad(270.), -np.deg2rad(45.), 0.])
    
elif args.view == 4: # lower left corner
    des_dir = folder_dir + '/view_4'
    os.system('mkdir -p ' + des_dir)
    
    cam_height = np.sqrt(2)/2 * r
    cam_dis = -np.sqrt(2)/2 * r
    
    camPos = np.array([0.+move_x, cam_height, cam_dis+move_z])
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