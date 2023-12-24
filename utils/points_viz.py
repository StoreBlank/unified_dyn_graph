import numpy as np
import cv2
import os

def viz_points_single_frame(img, points, cam_intr, cam_extr):
    
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
    point_size = 5
    point_color = (0, 0, 255)
    for k in range(points_proj.shape[0]):
        cv2.circle(img, (int(points_proj[k, 0]), int(points_proj[k, 1])), point_size,
                   point_color, -1)

    cv2.imwrite("point_viz.jpg", img)
 
        
if __name__ == "__main__":
    i = np.random.randint(0, 200)
    img_path = f'/mnt/sda/data/carrots/episode_0/camera_0/{i}_color.jpg'
    img = cv2.imread(img_path)
    
    eef_points_path = '/mnt/sda/data/carrots/episode_0/eef_pos.npy'
    eef_points = np.load(eef_points_path)
    eef_points = eef_points[i]
    # eef_points[1] -= 0.9
    eef_points = eef_points.reshape((1, 3))
    
    cam_intr_path = '/mnt/sda/data/carrots/camera_intrinsic_params.npy'
    cam_extr_path = '/mnt/sda/data/carrots/camera_extrinsic_matrix.npy'
    cam_intr, cam_extr = np.load(cam_intr_path), np.load(cam_extr_path)
    
    viz_points_single_frame(img, eef_points, cam_intr[0], cam_extr[0])
    
    