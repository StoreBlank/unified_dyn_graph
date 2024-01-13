import numpy as np
import cv2
import os
import glob
import PIL.Image as Image
import argparse
import open3d as o3d
import torch

from dgl.geometry import farthest_point_sampler
from utils_env import quaternion_to_rotation_matrix, fps_rad_idx

def merge_video(image_path, video_path):
    f_names = os.listdir(image_path)
    image_names = []
    for f_name in f_names:
        if '_color.jpg' in f_name:
            image_names.append(f_name)

    image_names.sort(key=lambda x: int(x.split('_')[0]))
        
    # print(image_names)

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    fps = 60

    img = Image.open(os.path.join(image_path, image_names[0]))

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, img.size)

    for img_name in image_names:
        img = cv2.imread(os.path.join(image_path, img_name))
        video_writer.write(img)

    print("Video merged!")

    video_writer.release()

def viz_one_point_single_frame(img, point, cam_intr, cam_extr, point_color=(0, 0, 255), point_size=1):
    
    # transform point
    num_point = 1 #point.shape[0]
    point_homo = np.concatenate([point, np.ones((num_point, 1))], axis=1) # (N, 4)
    point_homo = point_homo @ cam_extr.T # (N, 4)
    point_homo[:, 1] *= -1
    point_homo[:, 2] *= -1
    
    # project point
    fx, fy, cx, cy = cam_intr
    point_proj = np.zeros((point_homo.shape[0], 2))
    point_proj[:, 0] = point_homo[:, 0] * fx / point_homo[:, 2] + cx
    point_proj[:, 1] = point_homo[:, 1] * fy / point_homo[:, 2] + cy
    
    # visualize
    for k in range(point_proj.shape[0]):
        cv2.circle(img, (int(point_proj[k, 0]), int(point_proj[k, 1])), point_size,
                   point_color, -1)

    # cv2.imwrite("point_viz.jpg", img)
    return img

def viz_points_single_frame(img, points, cam_intr, cam_extr, group=0):
    
    point_colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
    ]
    
    for idx, point in enumerate(points):
        color = point_colors[group] 
        point = point.reshape((1, 3))
        img = viz_one_point_single_frame(img, point, cam_intr, cam_extr, color)
    
    return img

def viz_tool_one_point(episode_idx, data_dir, out_dir, cam_view=0):
    
    os.makedirs(out_dir, exist_ok=True)
    
    n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
    print(f"Episode {episode_idx} has {n_frames} frames.")
    tool_pos_path = os.path.join(data_dir, f"episode_{episode_idx}/tool_states.npy")
    tool_pos = np.load(tool_pos_path)
    tool_pos = tool_pos[:, :3]
    
    cam_intr_path = os.path.join(data_dir, "camera_intrinsic_params.npy")
    cam_extr_path = os.path.join(data_dir, "camera_extrinsic_matrix.npy")
    cam_intrs, cam_extrs = np.load(cam_intr_path), np.load(cam_extr_path)
    cam_intr, cam_extr = cam_intrs[cam_view], cam_extrs[cam_view]
    
    for i in range(n_frames):
        raw_img_path = os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_view}/{i}_color.jpg")
        raw_img = cv2.imread(raw_img_path)
        tool_points = tool_pos[i].reshape((1, 3))
        img = viz_one_point_single_frame(raw_img, tool_points, cam_intr, cam_extr)
        cv2.imwrite(os.path.join(out_dir, f"{i}_color.jpg"), img)
    
    # make video
    video_path = os.path.join(out_dir, f"episode_{episode_idx}.mp4")
    merge_video(out_dir, video_path)
    print(f"Video saved to {video_path}.")

def viz_tool_ptcl_one_frame(args, tool_names, tool_scale, tool_mesh_dir, sample_points=100, fps=True):
    # load image
    img_path = f'/mnt/sda/data/{args.data_name}/episode_{args.epi_idx}/camera_0/{args.idx}_color.jpg'
    img = cv2.imread(img_path)
    processed_img = img.copy()
    # load camera
    cam_intr_path = f'/mnt/sda/data/{args.data_name}/camera_intrinsic_params.npy'
    cam_extr_path = f'/mnt/sda/data/{args.data_name}/camera_extrinsic_matrix.npy'
    cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    
    n_tools = len(tool_names)
    points_list = []
    for i in range(n_tools):
        # convert mesh to point cloud
        tool_mesh_path = os.path.join(tool_mesh_dir, f'{tool_names[i]}.obj')
        tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
        
        tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 10000)
        # tool_surface = o3d.geometry.TriangleMesh.sample_points_uniformly(tool_mesh, sample_points)
        
        # scale the point cloud
        tool_surface.points = o3d.utility.Vector3dVector(np.asarray(tool_surface.points) * tool_scale[i])
        points_list.append(tool_surface)
            
        # load the pos and orientation of the tool
        tool_points_path = f'/mnt/sda/data/{args.data_name}/episode_{args.epi_idx}/{tool_names[i]}_states.npy' 
        tool_points = np.load(tool_points_path)
        
        tool_ori = tool_points[args.idx, 3:]
        tool_rot = quaternion_to_rotation_matrix(tool_ori)
        tool_surface.rotate(tool_rot)
        
        tool_pos = tool_points[args.idx, :3]
        tool_surface.translate(tool_pos)
        
        if args.vis:
            o3d.visualization.draw_geometries([tool_surface, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=tool_pos)])
        
        tool_surface_points = np.asarray(tool_surface.points)
        
        if fps:
            particle_tensor = torch.from_numpy(tool_surface_points).float().unsqueeze(0)
            fps_idx_tensor = farthest_point_sampler(particle_tensor, sample_points)[0]
            fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)
            # print(f'fps_idx_1: {fps_idx_1}')
            # fps_idx = fps_idx_1
            
            # downsample to uniform radius
            downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
            fps_radius = 0.15
            _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
            fps_idx_2 = fps_idx_2.astype(np.int32)
            fps_idx = fps_idx_1[fps_idx_2]
            print(f'tool {tool_names[i]} has {fps_idx.shape[0]} sample points.')
            
            # obtain fps tool surface points
            tool_surface_points = tool_surface_points[fps_idx]
            if args.vis:
                # visualize fps points
                fps_points = o3d.geometry.PointCloud()
                fps_points.points = o3d.utility.Vector3dVector(tool_surface_points)
                o3d.visualization.draw_geometries([fps_points, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=tool_pos)])
                
        
        # visualize points
        processed_img = viz_points_single_frame(processed_img, tool_surface_points, cam_intr[0], cam_extr[0], group=i)
    
    return processed_img
    
def viz_graph_one_frame(args, tool_names, tool_scale, tool_mesh_dir):
    # load camera
    cam_intr_path = f'/mnt/sda/data/{args.data_name}/camera_intrinsic_params.npy'
    cam_extr_path = f'/mnt/sda/data/{args.data_name}/camera_extrinsic_matrix.npy'
    cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    
    # load particles
    particles_path = f'/mnt/sda/data/{args.data_name}/episode_{args.epi_idx}/particles_pos.npy'
    particles = np.load(particles_path)
    particles_pos = particles[args.idx]
    print(f'frame {args.idx}  has {particles_pos.shape[0]} particles')
    
    # random sample particles
    obj_sample_points = 500
    obj_idx = np.random.choice(particles_pos.shape[0], obj_sample_points, replace=False)
    sample_particles_pos = particles_pos[obj_idx]
    # sample_particles_pos = particles_pos.copy()
    
    # draw tool particles
    processed_img = viz_tool_ptcl_one_frame(args, tool_names, tool_scale, tool_mesh_dir, vis=False)
    # draw obj particles
    processed_img = viz_points_single_frame(processed_img, sample_particles_pos, cam_intr[0], cam_extr[0], group=2)
    
    return processed_img

def viz_tools(args, data_dir, out_dir, tool_names, tool_scale, tool_mesh_dir, sample_points=100, fps=True):
    
    os.makedirs(out_dir, exist_ok=True)
    
    n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{args.epi_idx}/camera_0/*_color.jpg"))))
    print(f"Episode {args.epi_idx} has {n_frames} frames.")
    
    # load camera
    cam_intr_path = os.path.join(data_dir, 'camera_intrinsic_params.npy') 
    cam_extr_path = os.path.join(data_dir, 'camera_extrinsic_matrix.npy')
    cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    
    # get tool surface points
    n_tools = len(tool_names)
    tool_surface_points_list = []
    for i in range(n_tools):
        # convert mesh to point cloud
        tool_mesh_path = os.path.join(tool_mesh_dir, f'{tool_names[i]}.obj')
        tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
        
        tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 10000)
        # tool_surface = o3d.geometry.TriangleMesh.sample_points_uniformly(tool_mesh, sample_points)
        
        # scale the point cloud
        tool_surface.points = o3d.utility.Vector3dVector(np.asarray(tool_surface.points) * tool_scale[i])
        
        tool_surface_points = np.asarray(tool_surface.points)
        if fps:
            particle_tensor = torch.from_numpy(tool_surface_points).float().unsqueeze(0)
            fps_idx_tensor = farthest_point_sampler(particle_tensor, sample_points)[0]
            fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)
            
            # downsample to uniform radius
            downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
            fps_radius = 0.15
            _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
            fps_idx_2 = fps_idx_2.astype(np.int32)
            fps_idx = fps_idx_1[fps_idx_2]
            print(f'tool {tool_names[i]} has {fps_idx.shape[0]} sample points.')
            
            # obtain fps tool surface points
            tool_surface_points = tool_surface_points[fps_idx]
            if args.vis:
                # visualize fps points
                fps_points = o3d.geometry.PointCloud()
                fps_points.points = o3d.utility.Vector3dVector(tool_surface_points)
                o3d.visualization.draw_geometries([fps_points, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=tool_pos)])
        
        tool_surface_points_list.append(tool_surface_points)
            
    # translate and rotate the tool surface for each frame
    for i in range(2, n_frames):
        # load image
        img_path = os.path.join(data_dir, f"episode_{args.epi_idx}/camera_0/{i}_color.jpg")
        img = cv2.imread(img_path)
        
        for j in range(n_tools):
            # load tool surface
            tool_surface_i = o3d.geometry.PointCloud()
            tool_surface_i.points = o3d.utility.Vector3dVector(tool_surface_points_list[j])
        
            # load the pos and orientation of the tool
            tool_points_path = os.path.join(data_dir, f"episode_{args.epi_idx}/eef_states.npy")
            tool_points = np.load(tool_points_path)
        
            tool_ori = tool_points[i, 3:]
            tool_rot = quaternion_to_rotation_matrix(tool_ori)
            tool_surface_i.rotate(tool_rot)
        
            tool_pos = tool_points[i, :3]
            tool_surface_i.translate(tool_pos)
        
            if args.vis:
                o3d.visualization.draw_geometries([tool_surface_i, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=tool_pos)])
        
            tool_surface_i_points = np.asarray(tool_surface_i.points)
        
            # visualize points
            img = viz_points_single_frame(img, tool_surface_i_points, cam_intr[0], cam_extr[0], group=j)
        
        cv2.imwrite(os.path.join(out_dir, f"{i}_color.jpg"), img)
    
    # make video
    video_path = os.path.join(out_dir, f"episode_{args.epi_idx}.mp4")
    merge_video(out_dir, video_path)
    print(f"Video saved to {video_path}.")

def get_stats(points):
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    min_z, max_z = np.min(zs), np.max(zs)
    print(f'min_x: {min_x}, max_x: {max_x}')
    print(f'min_y: {min_y}, max_y: {max_y}')
    print(f'min_z: {min_z}, max_z: {max_z}')
    return ys

def viz_tool_2(args, data_dir, out_dir, tool_name, tool_scale, tool_mesh_dir, sample_points=100, fps=True):
    ### extract some points in the specifc areas of the tool
    
    os.makedirs(out_dir, exist_ok=True)
    
    n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{args.epi_idx}/camera_0/*_color.jpg"))))
    print(f"Episode {args.epi_idx} has {n_frames} frames.")
    
    # load camera
    cam_intr_path = os.path.join(data_dir, 'camera_intrinsic_params.npy') 
    cam_extr_path = os.path.join(data_dir, 'camera_extrinsic_matrix.npy')
    cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    
    # convert mesh to point cloud
    tool_mesh_path = os.path.join(tool_mesh_dir, f'{tool_name}.obj')
    tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
    
    tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 10000)
    # tool_surface = o3d.geometry.TriangleMesh.sample_points_uniformly(tool_mesh, sample_points)
    
    # scale the point cloud
    tool_surface.points = o3d.utility.Vector3dVector(np.asarray(tool_surface.points) * tool_scale)
    if args.vis:
        o3d.visualization.draw_geometries([tool_surface, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])])
    
    # filter points
    tool_points = np.asarray(tool_surface.points)
    print(tool_points.shape)
    
    # ys = get_stats(tool_points)
    # filter_idx = np.where(np.abs(ys) < 0.1)[0]
    # filtered_tool_points = tool_points[filter_idx]
    # print(filtered_tool_points.shape)
    
    ys = get_stats(tool_points)
    filter_idx = np.where((ys > -0.08) & (ys < -0.07))[0]
    filtered_tool_points = tool_points[filter_idx]
    print(filtered_tool_points.shape)
    ys = get_stats(filtered_tool_points)
    
    tool_surface.points = o3d.utility.Vector3dVector(filtered_tool_points)
    if args.vis:
        o3d.visualization.draw_geometries([tool_surface, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])])
    
    # fps
    tool_surface_points = np.asarray(tool_surface.points)
    if fps:
        particle_tensor = torch.from_numpy(tool_surface_points).float().unsqueeze(0)
        fps_idx_tensor = farthest_point_sampler(particle_tensor, sample_points)[0]
        fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)
        
        # downsample to uniform radius
        downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
        fps_radius = 0.1
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
        fps_idx_2 = fps_idx_2.astype(np.int32)
        fps_idx = fps_idx_1[fps_idx_2]
        print(f'tool {tool_name} has {fps_idx.shape[0]} sample points.')
        
        # obtain fps tool surface points
        tool_surface_points = tool_surface_points[fps_idx]
        if args.vis:
            # visualize fps points
            fps_points = o3d.geometry.PointCloud()
            fps_points.points = o3d.utility.Vector3dVector(tool_surface_points)
            o3d.visualization.draw_geometries([fps_points, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])])
            
    # translate and rotate the tool surface for each frame
    for i in range(2, n_frames):
        # load image
        img_path = os.path.join(data_dir, f"episode_{args.epi_idx}/camera_0/{i}_color.jpg")
        img = cv2.imread(img_path)
        
        # load tool surface
        tool_surface_i = o3d.geometry.PointCloud()
        tool_surface_i.points = o3d.utility.Vector3dVector(tool_surface_points)
    
        # load the pos and orientation of the tool
        tool_points_path = os.path.join(data_dir, f"episode_{args.epi_idx}/eef_states.npy")
        tool_points = np.load(tool_points_path)
    
        tool_ori = tool_points[i, 3:]
        tool_rot = quaternion_to_rotation_matrix(tool_ori)
        tool_surface_i.rotate(tool_rot)
    
        tool_pos = tool_points[i, :3]
        tool_surface_i.translate(tool_pos)
    
        if args.vis:
            o3d.visualization.draw_geometries([tool_surface_i, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=tool_pos)])
    
        tool_surface_i_points = np.asarray(tool_surface_i.points)
    
        # visualize points
        img = viz_points_single_frame(img, tool_surface_i_points, cam_intr[0], cam_extr[0], group=0)
        
        cv2.imwrite(os.path.join(out_dir, f"{i}_color.jpg"), img)
    
    # make video
    video_path = os.path.join(out_dir, f"episode_{args.epi_idx}.mp4")
    merge_video(out_dir, video_path)
    print(f"Video saved to {video_path}.")
    
    

#### visualization     
def viz_1(agrs):
    """viz points in one frame"""
    i = args.idx #np.random.randint(0, 200)
    data_name = args.data_name 
    epi_idx = args.epi_idx
    
    # load img
    img_path = f'/mnt/sda/data/{data_name}/episode_{epi_idx}/camera_0/{i}_color.jpg'
    img = cv2.imread(img_path)
    
    # load points
    tool = []
    
    tool_points_path = f'/mnt/sda/data/{data_name}/episode_{epi_idx}/tool_states.npy'
    tool_points = np.load(tool_points_path)
    tool_points = tool_points[i, :3]
    tool_points = tool_points.reshape((1, 3))
    print(f'frame {i} tool pos: {tool_points}')
    tool.append(tool_points)
    
    tool2_points_path = f'/mnt/sda/data/{data_name}/episode_{epi_idx}/tool_2_states.npy' 
    tool2_points = np.load(tool2_points_path)
    tool2_points = tool2_points[i, :3]
    tool2_points = tool2_points.reshape((1, 3))
    print(f'frame {i} tool 2 pos: {tool2_points}')
    tool.append(tool2_points)
    
    # load camera
    cam_intr_path = f'/mnt/sda/data/{data_name}/camera_intrinsic_params.npy'
    cam_extr_path = f'/mnt/sda/data/{data_name}/camera_extrinsic_matrix.npy'
    cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    # print(cam_intr)
    # print(cam_extr)
    
    processed_img = viz_points_single_frame(img, tool, cam_intr[0], cam_extr[0])
    out_dir = f'/mnt/sda/viz_tool/{data_name}'
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{i}_color.jpg"), processed_img)

def viz_2(args):
    data_name = args.data_name 
    epi_idx = args.epi_idx
    """viz tool center point"""
    data_dir = f'/mnt/sda/data/{data_name}'
    out_dir = f'/mnt/sda/viz_tool/{data_name}/{str(epi_idx).fill(3)}'
    viz_tool_one_point(epi_idx, data_dir, out_dir)

def viz_3(args):
    """viz tool surface in a frame"""
    data_root = f'/mnt/sda/data/{args.data_name}'
    tool_mesh_dir = os.path.join(data_root, 'geometries/tools')
    tool_names = ['sponge']
    tool_scale = [8.0]
    
    processed_img = viz_tool_ptcl_one_frame(args, tool_names, tool_scale, tool_mesh_dir)
    # processed_img = viz_graph_one_frame(args, tool_names, tool_scale, tool_mesh_dir)
    
    out_dir = f'/mnt/sda/viz_tool/{args.data_name}'
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{args.idx}_tool.jpg"), processed_img)

def viz_4(args):
    """viz tool frames"""
    data_root = f'/mnt/sda/data/{args.data_name}'
    tool_mesh_dir = os.path.join(data_root, 'geometries/tools')
    tool_names = ['sponge']
    tool_scale = [8.0]

    out_dir = f'/mnt/sda/viz_tool/{args.data_name}/{args.epi_idx}'
    # viz_tools(args, data_root, out_dir, tool_names, tool_scale, tool_mesh_dir)
    viz_tool_2(args, data_root, out_dir, tool_names[0], tool_scale[0], tool_mesh_dir)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='granular_sweeping')
    parser.add_argument('--epi_idx', type=int, default=900)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--vis', type=bool, default=False)
    args = parser.parse_args()
    
    # args.idx = np.random.randint(0, 200)
    
    ### viz points in one frame
    # viz_1(args)
    
    ### viz tool center point
    # viz_2(args)
    
    ### viz tool surface in a frame
    # viz_3(args)
    
    ### viz tool frames
    viz_4(args)
    