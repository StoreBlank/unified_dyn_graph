import numpy as np
import cv2
import os
import glob
import PIL.Image as Image
import argparse
import open3d as o3d

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

def viz_one_point_single_frame(img, point, cam_intr, cam_extr, point_color=(0, 0, 255), point_size=3):
    
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

def viz_points_single_frame(img, points, cam_intr, cam_extr):
    
    point_colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
    ]
    for idx, point in enumerate(points):
        color = point_colors[0] #point_colors[idx%len(point_colors)]
        point = point.reshape((1, 3))
        print(f"idx: {idx}, point {point}")
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

def viz_tool_ptcl_one_frame():
    
    pass

    

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

def viz_3(args, vis=False):
    """viz tool surface in a frame"""
    data_root = f'/mnt/sda/data/{args.data_name}'
    tool_mesh_dir = os.path.join(data_root, 'geometries/tools')
    tool_names = os.listdir(tool_mesh_dir) # ['dustpan.obj', 'sponge_2.obj']
    tool_scale = {
        'dustpan': 1.1,
        'sponge_2': 8.0,
    }
    
    # convert mesh to point cloud
    tool_list = []
    tool_name = tool_names[0]
    tool_mesh_path = os.path.join(tool_mesh_dir, tool_name)
    tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
    tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 100)
    # scale the point cloud
    scale_factor = 1.1
    tool_surface.points = o3d.utility.Vector3dVector(np.asarray(tool_surface.points) * scale_factor)
    # tool_list.append((tool_mesh, tool_surface))
    if vis:
        # o3d.visualization.draw_geometries([tool_surface])
        # visualize the world coordinate frame
        o3d.visualization.draw_geometries([tool_surface, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)])
        
    # load the pos of the tool
    tool2_points_path = f'/mnt/sda/data/{args.data_name}/episode_{args.epi_idx}/tool_2_states.npy' 
    tool2_points = np.load(tool2_points_path)
    tool2_points = tool2_points[args.idx, :3]
    tool2_points = tool2_points.reshape((1, 3))
    print(f'frame {args.idx} tool 2 pos: {tool2_points}')
    
    # move the tool surface to the pos of the tool
    tool_surface.translate(tool2_points[0])
    if vis:
        o3d.visualization.draw_geometries([tool_surface, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=tool2_points[0])])
    
    tool_surface_points = np.asarray(tool_surface.points)
    print(f"tool surface points: {tool_surface_points.shape}")
    
    # load image
    img_path = f'/mnt/sda/data/{args.data_name}/episode_{args.epi_idx}/camera_0/{args.idx}_color.jpg'
    img = cv2.imread(img_path)
    # load camera
    cam_intr_path = f'/mnt/sda/data/{args.data_name}/camera_intrinsic_params.npy'
    cam_extr_path = f'/mnt/sda/data/{args.data_name}/camera_extrinsic_matrix.npy'
    cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    
    # visualize points
    processed_img = viz_points_single_frame(img, tool_surface_points, cam_intr[0], cam_extr[0])
    out_dir = f'/mnt/sda/viz_tool/{args.data_name}'
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{args.idx}_tool.jpg"), processed_img)
    
    
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='granular_sweeping_dustpan')
    parser.add_argument('--epi_idx', type=int, default=0)
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    
    # i = args.idx #np.random.randint(0, 200)
    # data_name = args.data_name 
    # epi_idx = args.epi_idx
    
    ### viz points in one frame
    # viz_1(args)
    
    ### viz tool center point
    # viz_2(args)
    
    ### viz tool surface in a frame
    viz_3(args)
    