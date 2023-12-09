import cv2
import os

def video_to_frames(video_file, output_dir):
    # Capture the video from the file
    vidcap = cv2.VideoCapture(video_file)

    # Check if the video opened successfully
    if not vidcap.isOpened():
        print("Error: Could not open video.")
        return

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    success, image = vidcap.read()
    while success:
        # Save the current frame
        cv2.imwrite(os.path.join(output_dir, f"frame{count:04d}.jpg"), image)     

        # Read the next frame from the video
        success, image = vidcap.read()
        count += 1
    
    print(f"Saved {count} frames to {output_dir}.")

### video to frames
video_list = [0]
for i in video_list:
    video_path = f"/home/baoyu/2023/unified_dyn_graph/videos/rigid_cloth_{i}.mp4"
    output_dir = f"/home/baoyu/2023/unified_dyn_graph/videos/frames/rigid_cloth_{i}"
    video_to_frames(video_path, output_dir)
