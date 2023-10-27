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
    for n in range(1):
        i = 1
        epi_path = f"/media/baoyu/sumsung/debug/Tshirt/episode_{i}/camera_0"
        image_path = f"/media/baoyu/sumsung/debug/Tshirt/episode_{i}/camera_0"
        video_path = f"/media/baoyu/sumsung/video/Tshirt/video_{i}.mp4" 
        merge_video(image_path, video_path)
        # open json file
        # with open(os.path.join(epi_path, 'property.json'), 'r') as f:
        #     property = json.load(f)
        # print(i, property['cluster_spacing'])