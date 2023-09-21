import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
import copy

from dgl.geometry import farthest_point_sampler
import open3d as o3d

from utils import fps, depth2fgpcd, pcd2pix, fps_np, downsample_pcd

env = 'carrots'
folder_path = f'ptcl_data/{env}'

pcd_all_list = []
extrinsic_matrixs = []
global_scale = 24

for i in range(1, 5):
    # load data
    cam_view = 'view_{}'.format(i)
    dir_path = os.path.join(folder_path, cam_view)
    
    # obs = np.load(os.path.join(dir_path, 'obs.npy'))
    camera_intrinsic_params = np.load(os.path.join(dir_path, 'camera_intrinsic_params.npy')) # [fx, fy, cx, cy]
    camera_extrinsic_matrix = np.load(os.path.join(dir_path, 'camera_extrinsic_matrix.npy'))
    print('i:', i)
    print('camera_intrinsic_params\n', camera_intrinsic_params)
    print('camera_extrinsic_matrix\n', camera_extrinsic_matrix)
    extrinsic_matrixs.append(camera_extrinsic_matrix)
    
    pcd = o3d.io.read_point_cloud(os.path.join(dir_path, 'fgpcd.pcd'))
    pcd_array = np.asarray(pcd.points)
    print(pcd_array.shape, pcd_array.mean(axis=0))
    
    pcd_all_list.append(pcd)

o3d.visualization.draw_geometries(pcd_all_list)

# pcd_all = o3d.geometry.PointCloud()
# pcd_all = [pcd_all_list[0]]
# for point_id in range(2):
#     trans = extrinsic_matrixs[point_id] @ np.linalg.inv(extrinsic_matrixs[0])
#     pcd_all_list[point_id].transform(trans)
#     pcd_all.append(pcd_all_list[point_id])
# 
# o3d.visualization.draw_geometries(pcd_all)

# o3d.visualization.draw_geometries([pcd_all_list[1]])