import numpy as np
import open3d as o3d

from data_generation.utils import fps_with_idx

def pos2pcd(particle_pos, viz = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(particle_pos)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    return pcd

frame_id = 100
pos_path = '/mnt/sda/data/carrots_5/episode_0/particles_pos.npy'
particles_pos = np.load(pos_path)
particle_pos = particles_pos[frame_id]
print('particle pos shape: ', particle_pos.shape)

pcd = pos2pcd(particle_pos, viz = True)
print('num of points: ', len(pcd.points))
print('points: ', np.asarray(pcd.points))

