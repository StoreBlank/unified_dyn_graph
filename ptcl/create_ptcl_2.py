import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
import open3d as o3d

from utils import depth2fgpcd, depth2fgpcd_top, opengl2cam

env = 'shirt'
views = [1, 2, 3, 4]

for view in views:
    dir_path = f'data/{env}/camera_{str(view)}'
    epi_path = os.path.join(dir_path, 'episode_0')

    raw_obs = np.load(os.path.join(epi_path, '0_obs.npy'))

    camera_intrinsic_params = np.load(os.path.join(dir_path, 'camera_intrinsic_params.npy'))
    camera_ext_matrix = np.load(os.path.join(dir_path, 'camera_extrinsic_matrix.npy'))

    global_scale = 1
    obs = raw_obs
    depth = obs[..., -1] / global_scale
    color = obs[..., :3][..., ::-1] / global_scale

    def depth2fgpcd_new(depth, intr, extr):
        h, w = depth.shape
        fx, fy, cx, cy = intr
        rot = extr[:3, :3]
        trans = extr[:3, 3]
        
        # get inverse transformation
        inv_rot = np.linalg.inv(rot)
        inv_extr = np.eye(4)
        inv_extr[:3, :3] = inv_rot
        inv_extr[:3, 3] = - inv_rot @ trans
        
        pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
        fgpcd = np.zeros((depth.shape[0], depth.shape[1], 3))

        # transform depth (in image coordinates) to fgpcd (in camera coordinates)
        fgpcd[:, :, 0] = (pos_x - cx) * depth / fx
        fgpcd[:, :, 1] = (pos_y - cy) * depth / fy
        fgpcd[:, :, 2] = depth

        # transform fgpcd (in camera coordinates) to world coordinates
        fgpcd_world = np.matmul(inv_extr, np.concatenate([fgpcd.reshape(-1, 3), np.ones((fgpcd.reshape(-1, 3).shape[0], 1))], axis=1).T).T[:, :3]

        # the up direction is +Y axis, so to filter out the table we need to filter out small Y values
        mask = fgpcd_world[..., 1] > (fgpcd_world[..., 1].min() + 0.51) # Tshirt 0.51
        
        fgpcd_world = fgpcd_world[mask]
        return inv_extr, fgpcd_world

    # as stated in test_carrots.py, 
    # although pyflex uses OpenGL convention (in which -Z is pointing away from the camera),
    # it actually uses the Open3D version of the extrinsic matrix for projection.
    # In other words, the larger the depth values are, the further the points are from the camera.
    # Therefore, we need to use the Open3D version of the extrinsic matrix to transform the points 
    # from camera coordinates back to world coordinates.
    ogl_to_o3d = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    camera_ext_matrix_o3d = ogl_to_o3d @ camera_ext_matrix
    inv_extr, fgpcd = depth2fgpcd_new(depth, camera_intrinsic_params, camera_ext_matrix_o3d)
    print(inv_extr)

    pcd = o3d.geometry.PointCloud()

    print(fgpcd.mean(0))
    print(fgpcd.min(0))
    print(fgpcd.max(0))

    pcd.points = o3d.utility.Vector3dVector(fgpcd)
    o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud(os.path.join('ptcl_data_2', 'fgpcd_%d.pcd' % view), pcd)
