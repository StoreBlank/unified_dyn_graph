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
    fps = 40

    img = Image.open(os.path.join(image_path, image_names[0]))

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, img.size)

    for img_name in image_names:
        img = cv2.imread(os.path.join(image_path, img_name))
        video_writer.write(img)

    print("Video merged!")

    video_writer.release()

if __name__ == '__main__':
    # epi_num = 10
    # for n in range(epi_num):
    #     #i = np.random.randint(0, 100)
    #     i = n
    #     #j = np.random.randint(0, 4)
    #     j = 0
    #     epi_path = f"/media/baoyu/sumsung/rope/episode_{i}/camera_{j}"
    #     image_path = f"/media/baoyu/sumsung/rope/episode_{i}/camera_{j}"
    #     video_path = f"/media/baoyu/sumsung/video/rope/video_{i}.mp4" 
    #     merge_video(image_path, video_path)
    
    # for i in range(1):
    #     image_path = f"/data_dense/granular_scooping/episode_0/camera_{i}"
    #     video_path = f"videos/bowl_granular_{i}.mp4" 
    #     merge_video(image_path, video_path)

    for i in range(1):
        image_path = f"data_dense/granular_scooping/camera_{i}"
        video_path = f"videos/granular_scooping_4.mp4" 
        merge_video(image_path, video_path)
