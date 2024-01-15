import os
import shutil
import random

# Path to the parent directory containing "folder_0" to "folder_5"
parent_dir = '/mnt/sda/data/rope_stiffness'

# Path to the big folder where all subfolders will be merged
big_folder = '/mnt/sda/data/rope_stiffness/rope_stiff'

# Ensure the big folder exists
os.makedirs(big_folder, exist_ok=True)

# List for storing paths of all subfolders
all_subfolders = []

# Loop through each main folder and collect subfolder paths
for i in range(6):  # for folder_0 to folder_5
    folder_name = f'rope_{i}'
    folder_path = os.path.join(parent_dir, folder_name)

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            all_subfolders.append(subfolder_path)

print(f"Found {len(all_subfolders)} subfolders")

# Shuffle the list of subfolders
random.shuffle(all_subfolders)

# Copy and rename subfolders to the big folder
for i, subfolder_path in enumerate(all_subfolders):
    new_subfolder_name = f'episode_{i}'
    new_subfolder_path = os.path.join(big_folder, new_subfolder_name)
    shutil.copytree(subfolder_path, new_subfolder_path)
    print(f"Subfolder {subfolder_path} copied to {new_subfolder_path}")

print(f"All subfolders have been merged and renamed in {big_folder}")
