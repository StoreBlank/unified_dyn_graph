import os
import numpy as np
import cv2
import PIL.Image as Image
import json

def merge_video(image_path, video_path):
    f_names = os.listdir(image_path)
    image_names = []
    for f_name in f_names:
        if '_color.jpg' in f_name:
            image_names.append(f_name)

    image_names.sort(key=lambda x: int(x.split('_')[0]))
        
    # print(image_names)

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    fps = 10

    img = Image.open(os.path.join(image_path, image_names[0]))

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, img.size)

    for img_name in image_names:
        img = cv2.imread(os.path.join(image_path, img_name))
        video_writer.write(img)

    print("Video merged!")

    video_writer.release()

if __name__ == '__main__':
    # epi_start = 10
    # epi_num = 20
    # # l = np.random.choice(100, epi_num, replace=False)
    # # print(l)
    # for n in range(epi_start, epi_num):
    #     i = n
    #     # i = l[n]
    #     #j = np.random.randint(0, 4)
    #     j = 0
    #     epi_path = f"/mnt/sda/data/rope/episode_{i}/camera_{j}"
    #     image_path = f"/mnt/sda/data/rope/episode_{i}/camera_{j}"
    #     video_dir = f"/mnt/sda/videos/rope" 
    #     video_path = f"/mnt/sda/videos/rope/video_{i}.mp4" 
    #     os.makedirs(video_dir, exist_ok=True)
    #     merge_video(image_path, video_path)
    
    for n in [0, 1]:
        i = n
        j = 0
        epi_path = f"/mnt/sda/data/rope_stiff/rope_8/rope/episode_{i}/camera_{j}"
        image_path = f"/mnt/sda/data/rope_stiff/rope_8/rope/episode_{i}/camera_{j}"
        video_dir = f"/mnt/sda/videos/rope_stiff" 
        video_path = os.path.join(video_dir, f"rope_{i}.mp4") 
        os.makedirs(video_dir, exist_ok=True)
        merge_video(image_path, video_path)
    
    
