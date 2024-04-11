import os
import numpy as np
import cv2
import PIL.Image as Image
import json
import argparse

def merge_video(image_path, video_path, fps=20):
    f_names = os.listdir(image_path)
    image_names = []
    for f_name in f_names:
        # print(f_name)
        if f_name.endswith('.jpg'):
            image_names.append(f_name)
    
    image_names.sort(key=lambda x: int(x.split('_')[0]))

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')

    img = Image.open(os.path.join(image_path, image_names[0]))

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, img.size)

    for img_name in image_names:
        img = cv2.imread(os.path.join(image_path, img_name))
        video_writer.write(img)

    print("Video merged!")

    video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epi", type=int, default=0)
    args = parser.parse_args()
    
    # epi_start = 45
    # epi_num = 46
    # # l = np.random.choice(100, epi_num, replace=False)
    # # print(l)
    # for n in range(epi_start, epi_num):
    #     i = n
    #     j = 0
    #     image_path = f"/mnt/sda/data/cloth/episode_{i}/camera_{j}"
    #     video_dir = f"/mnt/sda/videos/" 
    #     video_path =  os.path.join(video_dir, f"video_{i}.mp4")
    #     os.makedirs(video_dir, exist_ok=True)
    #     merge_video(image_path, video_path)
    
    data_name = "granular/carrots_flat"
    epi_idx = args.epi
    image_path = f"/mnt/sda/data/{data_name}/episode_{epi_idx}/camera_0"
    video_dir = f"/mnt/sda/videos/{data_name}" 
    video_path =  os.path.join(video_dir, f"video_{epi_idx}.mp4")
    os.makedirs(video_dir, exist_ok=True)
    merge_video(image_path, video_path)
    
    # for n in range(1):
    #     i = n
    #     name = "0.2"
    #     image_path = f"/mnt/sda/data/granular_size/carrots/{name}/camera_0"
    #     video_dir = f"/mnt/sda/videos/granular_size/1" 
    #     video_path = os.path.join(video_dir, f"{name}.mp4") 
    #     os.makedirs(video_dir, exist_ok=True)
    #     merge_video(image_path, video_path)
    
    # for i in range(4):
    #     epi_path = f"/mnt/sda/data/box_com/episode_{i:03d}/images"
    #     video_path = f"/mnt/sda/videos/box_com/video_{i:03d}.mp4"
    #     os.makedirs(os.path.dirname(video_path), exist_ok=True)
    #     merge_video(epi_path, video_path)
    
    
