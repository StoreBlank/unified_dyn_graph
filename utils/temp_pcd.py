import numpy as np

from data_generation.utils import pos2pcd, fps_to_pos, fps_with_idx

pos_path = '/media/baoyu/sumsung/rope/episode_0/camera_0/0_particles.npy'
particle_pos = np.load(pos_path)
particle_pos = particle_pos.reshape(-1, 4)[:, :3]
print('particle pos shape: ', particle_pos.shape)
# print('particle pos: ', particle_pos)

pcd = pos2pcd(particle_pos, viz = True)
print('num of points: ', len(pcd.points))
# print('points: ', np.asarray(pcd.points))

# particle_num_down = 100
# downsampled_pos, dist = fps_to_pos(pcd, particle_num_down, viz = True)
# print('downsampled pos shape: ', downsampled_pos.shape)
# print('dist: ', dist)

sampled_particle_pos, sampled_idx = fps_with_idx(particle_pos, 10000)
print('sampled particle pos shape: ', sampled_particle_pos.shape)
print('sampled idx: ', sampled_idx)
pcd_down = pos2pcd(sampled_particle_pos, viz = True)
