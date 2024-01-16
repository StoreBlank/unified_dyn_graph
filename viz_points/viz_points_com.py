import os
import numpy as np
import cv2
import glob

def draw_points(img, points, point_size=5, point_color=(0,0,255)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), point_size, point_color, -1)
    return img

def convert_coordinates(point, screen_height=720):
        return np.array([point[0], screen_height - point[1]])

def viz_point_com(data_dir, out_dir, epi_idx):
    out_dir = os.path.join(out_dir, f"{epi_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)
    
    data_dir = os.path.join(data_dir, f"episode_{epi_idx:03d}")
    img_data_dir = os.path.join(data_dir, "images")
    box_states = np.load(os.path.join(data_dir, "box_states.npy"))
    eef_states = np.load(os.path.join(data_dir, "eef_states.npy"))
    num_frames = len(list(glob.glob(os.path.join(img_data_dir, "*.png"))))
    assert num_frames == len(box_states) == len(eef_states)
    
    for i in range(num_frames):
        img_path = os.path.join(img_data_dir, f"{i:03d}.png")
        img = cv2.imread(img_path)
        
        # draw box points
        box_state = box_states[i]
        box_pos = convert_coordinates(box_state[:2]).reshape((1, 2))
        img = draw_points(img, box_pos) # red
        
        # draw eef points
        eef_pos = convert_coordinates(eef_states[i]).reshape((1,2))
        img = draw_points(img, eef_pos, point_color=(0, 255, 0)) # green
    
        # save image
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}.png"), img)

def viz_com(data_dir, out_dir, epi_idx):
    out_dir = os.path.join(out_dir, f"{epi_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)
    
    data_dir = os.path.join(data_dir, f"episode_{epi_idx:03d}")
    img_data_dir = os.path.join(data_dir, "images")
    
    box_com = np.load(os.path.join(data_dir, "box_com.npy"))
    box_size = box_com[0]
    com = box_com[1]
    print(f"Episode {epi_idx}, box size: {box_size}, com: {com}")
    com[1] *= -1
    
    box_states = np.load(os.path.join(data_dir, "box_states.npy"))
    num_frames = len(list(glob.glob(os.path.join(img_data_dir, "*.png"))))
    assert num_frames == len(box_states)
    
    for i in range(num_frames):
        img_path = os.path.join(img_data_dir, f"{i:03d}.png")
        img = cv2.imread(img_path)
        
        # draw box points
        box_state = box_states[i]
        box_pos = convert_coordinates(box_state[:2])
        box_rad = box_state[2]
        print(f"box position {box_pos}, theta: {box_rad}")
        img = draw_points(img, box_pos.reshape((1,2)))
        
        # rotate the center of mass
        com_rotated_x = com[0] * np.cos(box_rad) + com[1] * np.sin(box_rad)
        com_rotated_y = -com[0] * np.sin(box_rad) + com[1] * np.cos(box_rad)
        
        # translate back to the world coordinate system
        com_x = box_pos[0] + com_rotated_x
        com_y = box_pos[1] + com_rotated_y
        com_pos = np.array([com_x, com_y])
        print(f"com position {com_pos}")
        img = draw_points(img, com_pos.reshape((1,2)), point_color=(0, 255, 0)) # green
        
        # save image
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}.png"), img)
        
        
if __name__ == "__main__":
    data_dir = "/mnt/sda/data/box"
    out_dir = "/mnt/sda/viz_eef/box_com"
    epi_idx = 0
    # viz_point_com(data_dir, out_dir, epi_idx)
    viz_com(data_dir, out_dir, epi_idx)
        
        
    
    
    
    