import numpy as np
import pyflex
import open3d as o3d
import torch
from dgl.geometry import farthest_point_sampler

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

def get_camera_intrinsics(screenHeight, screenWidth):
    projMat = pyflex.get_projMatrix().reshape(4, 4).T 
    cx = screenWidth / 2.0
    cy = screenHeight / 2.0
    fx = projMat[0, 0] * cx
    fy = projMat[1, 1] * cy
    camera_intrinsic_params = np.array([fx, fy, cx, cy])
    return camera_intrinsic_params

def get_camera_extrinsics():
    return pyflex.get_viewMatrix().reshape(4, 4).T


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

def pos2pcd(particle_pos, viz = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(particle_pos)
    
    if viz:
        o3d.visualization.draw_geometries([pcd])
    
    return pcd
    
def fps_to_pos(pcd, particle_num, init_idx = -1, viz = False):
    """
    Input: 
        pcd: o3d.geometry.PointCloud() 
        particle_num: int sample number
        init_idx: int index of the first point
        viz: bool visualize the sampled points
    Output:
        pcd_fps: np.array() sampled points
        dist.max(): float the maximum distance between the sampled points and the original points
    """
    n_points = len(pcd.points)
    if n_points < particle_num:
        pcd = np.asarray(pcd.points)
        return pcd, 0
    else:
        pcd = np.asarray(pcd.points)
        pcd_tensor = torch.from_numpy(pcd).float()[None, ...]
        if init_idx == -1:
            # init_idx = findClosestPoint(pcd, pcd.mean(axis=0))
            pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num)[0]
        else:
            pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num, init_idx)[0]
        pcd_fps_tensor = pcd_tensor[0, pcd_fps_idx_tensor]
        pcd_fps = pcd_fps_tensor.numpy()
        dist = np.linalg.norm(pcd[:, None] - pcd_fps[None, :], axis=-1)
        dist = dist.min(axis=1)
        if viz:
            pcd_fps_pcd = o3d.geometry.PointCloud()
            pcd_fps_pcd.points = o3d.utility.Vector3dVector(pcd_fps)
            o3d.visualization.draw_geometries([pcd_fps_pcd])
        return pcd_fps, dist.max()
    
def fps_with_idx(points, N):
    """
    Input:
        points: np.array() particle positions
        N: int sample number
    Output:
        points[farthest_pts_idx]: np.array() sampled points
        farthest_pts_idx: np.array() indices of the sampled points
    """
    if N > len(points):
        return points, np.arange(len(points))
    else:
        # start with the first point
        farthest_pts_idx = [0]
        distances = np.full(len(points), np.inf)
        
        for _ in range(1, N):
            last_point = points[farthest_pts_idx[-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            farthest_pts_idx.append(np.argmax(distances))
            
        return points[farthest_pts_idx], np.array(farthest_pts_idx)
        