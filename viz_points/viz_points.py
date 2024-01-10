import numpy as np
import cv2
import os
import glob
import PIL.Image as Image
import argparse

from utils_env import quaternion_to_rotation_matrix

def merge_video(image_path, video_path):
    f_names = os.listdir(image_path)
    image_names = []
    for f_name in f_names:
        if '_color.jpg' in f_name:
            image_names.append(f_name)

    image_names.sort(key=lambda x: int(x.split('_')[0]))
        
    # print(image_names)

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    fps = 20

    img = Image.open(os.path.join(image_path, image_names[0]))

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, img.size)

    for img_name in image_names:
        img = cv2.imread(os.path.join(image_path, img_name))
        video_writer.write(img)

    print("Video merged!")

    video_writer.release()

def viz_points_single_frame(img, points, cam_intr, cam_extr):
    
    # TODO: calibrate the tool point
    # points[:, 2] -= 0.02
    
    # transform points
    num_points = points.shape[0]
    points_homo = np.concatenate([points, np.ones((num_points, 1))], axis=1) # (N, 4)
    points_homo = points_homo @ cam_extr.T # (N, 4)
    points_homo[:, 1] *= -1
    points_homo[:, 2] *= -1
    
    # project points
    fx, fy, cx, cy = cam_intr
    points_proj = np.zeros((points_homo.shape[0], 2))
    points_proj[:, 0] = points_homo[:, 0] * fx / points_homo[:, 2] + cx
    points_proj[:, 1] = points_homo[:, 1] * fy / points_homo[:, 2] + cy
    
    # visualize
    point_size = 1
    point_color = (0, 0, 255)
    for k in range(points_proj.shape[0]):
        cv2.circle(img, (int(points_proj[k, 0]), int(points_proj[k, 1])), point_size,
                   point_color, -1)

    # cv2.imwrite("point_viz.jpg", img)
    return img

def viz_eef(episode_idx, data_dir, out_dir, cam_view=0):
    
    os.makedirs(out_dir, exist_ok=True)
    
    n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
    print(f"Episode {episode_idx} has {n_frames} frames.")
    eef_pos_path = os.path.join(data_dir, f"episode_{episode_idx}/eef_states.npy")
    eef_states = np.load(eef_pos_path)
    
    cam_intr_path = os.path.join(data_dir, "camera_intrinsic_params.npy")
    cam_extr_path = os.path.join(data_dir, "camera_extrinsic_matrix.npy")
    cam_intrs, cam_extrs = np.load(cam_intr_path), np.load(cam_extr_path)
    cam_intr, cam_extr = cam_intrs[cam_view], cam_extrs[cam_view]
    
    for i in range(n_frames):
        raw_img_path = os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_view}/{i}_color.jpg")
        raw_img = cv2.imread(raw_img_path)

        eef_pos = eef_states[i][0:3]
        eef_ori = eef_states[i][6:10]
        eef_rot_mat = quaternion_to_rotation_matrix(eef_ori)
        eef_final_pos = eef_pos + np.dot(eef_rot_mat, np.array([0., 0., 1.0])).reshape((1, 3))
        
        img = viz_points_single_frame(raw_img, eef_final_pos, cam_intr, cam_extr)
        cv2.imwrite(os.path.join(out_dir, f"{i}_color.jpg"), img)
    
    # make video
    video_path = os.path.join(out_dir, f"episode_{episode_idx}.mp4")
    merge_video(out_dir, video_path)
    print(f"Video saved to {video_path}.")

def viz_granular_eef(episode_idx, data_dir, out_dir, cam_view=0):
    
    os.makedirs(out_dir, exist_ok=True)
    
    n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
    print(f"Episode {episode_idx} has {n_frames} frames.")
    eef_pos_path = os.path.join(data_dir, f"episode_{episode_idx}/eef_states.npy")
    eef_states = np.load(eef_pos_path)
    
    cam_intr_path = os.path.join(data_dir, "camera_intrinsic_params.npy")
    cam_extr_path = os.path.join(data_dir, "camera_extrinsic_matrix.npy")
    cam_intrs, cam_extrs = np.load(cam_intr_path), np.load(cam_extr_path)
    cam_intr, cam_extr = cam_intrs[cam_view], cam_extrs[cam_view]
    
    # x_max: +-0.51
    # h_max: 1.26
    # thickness: 0.09
    n_eef_points = 5
    h = 1.25
    z = 0.045
    eef_point_pos = np.array([
        [0.5, z, h],
        [-0.5, z, h],
        [0, z, h],
        [0.25, z, h],
        [-0.25, z, h]
    ])
    for i in range(n_frames):
        raw_img_path = os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_view}/{i}_color.jpg")
        raw_img = cv2.imread(raw_img_path)
        
        for j in range(n_eef_points):
            eef_pos = eef_states[i][0:3]
            eef_ori = eef_states[i][6:10]
            eef_rot_mat = quaternion_to_rotation_matrix(eef_ori)
            eef_final_pos = eef_pos + np.dot(eef_rot_mat, eef_point_pos[j]).reshape((1, 3))
            
            img = viz_points_single_frame(raw_img, eef_final_pos, cam_intr, cam_extr)
        
        cv2.imwrite(os.path.join(out_dir, f"{i}_color.jpg"), img)
    
    # make video
    video_path = os.path.join(out_dir, f"episode_{episode_idx}.mp4")
    merge_video(out_dir, video_path)
    print(f"Video saved to {video_path}.")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='carrots')
    parser.add_argument('--epi_idx', type=int, default=0)
    parser.add_argument('--idx', type=int, default=3)
    args = parser.parse_args()
    
    i = args.idx #np.random.randint(0, 200)
    data_name = args.data_name 
    epi_idx = args.epi_idx
    
    # img_path = f'/mnt/sda/data/{data_name}/episode_{epi_idx}/camera_0/{i}_color.jpg'
    # img = cv2.imread(img_path)
    
    # eef_points_path = f'/mnt/sda/data/{data_name}/episode_{epi_idx}/eef_pos.npy'
    # eef_points = np.load(eef_points_path)
    # eef_points = eef_points[i][:3]
    # eef_points = eef_points.reshape((1, 3))
    # print(f'frame {i} eef pos: {eef_points}')
    
    # cam_intr_path = f'/mnt/sda/data/{data_name}/camera_intrinsic_params.npy'
    # cam_extr_path = f'/mnt/sda/data/{data_name}/camera_extrinsic_matrix.npy'
    # cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    
    # img = viz_points_single_frame(img, eef_points, cam_intr[0], cam_extr[0])
    
    
    data_dir = f'/mnt/sda/data/{data_name}'
    out_dir = f'/mnt/sda/viz_eef/{data_name}'
    # viz_eef(epi_idx, data_dir, out_dir)
    viz_granular_eef(epi_idx, data_dir, out_dir)
    
    