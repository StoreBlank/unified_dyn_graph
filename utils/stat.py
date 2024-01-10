import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

def get_steps(data_dir, epi_idx):
    steps_path = os.path.join(data_dir, f"episode_{epi_idx}/steps.npy")
    steps = np.load(steps_path)
    print(f"Episode {epi_idx} steps: {steps}")

def get_eef_pos(data_dir, epi_idx):
    eef_pos_path = os.path.join(data_dir, f"episode_{epi_idx}/eef_pos.npy")
    eef_pos = np.load(eef_pos_path)
    print(f"Episode {epi_idx} eef pos: {eef_pos.shape}")

def get_rope_property_params(data_dir, epi_start, epi_end):
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        print(f'Episode {i}, thickness: {property_params["thickness"]}, friction: {property_params["dynamic_friction"]}, stiffness: {property_params["cluster_spacing"]}')

def get_rope_normalized_property_params(data_dir, epi_start, epi_end):
    thickness_list = []
    friction_list = []
    stiffness_list = []
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        thickness_list.append(property_params['thickness'])
        friction_list.append(property_params['dynamic_friction'])
        stiffness_list.append(property_params['cluster_spacing'])
    
    # normalization
    thickness_list, friction_list, stiffness_list = np.array(thickness_list), np.array(friction_list), np.array(stiffness_list)
    thickness_list = (thickness_list - np.min(thickness_list)) / (np.max(thickness_list) - np.min(thickness_list))
    friction_list = (friction_list - np.min(friction_list)) / (np.max(friction_list) - np.min(friction_list))
    stiffness_list = (stiffness_list - np.min(stiffness_list)) / (np.max(stiffness_list) - np.min(stiffness_list))
    
    print(f'thickness: min: {np.min(thickness_list)}, max: {np.max(thickness_list)}')
    print(f'friction: min: {np.min(friction_list)}, max: {np.max(friction_list)}')
    print(f'stiffness: min: {np.min(stiffness_list)}, max: {np.max(stiffness_list)}')
        

def get_rope_property_stat(data_dir, out_dir, epi_start, epi_end):
    
    os.makedirs(out_dir, exist_ok=True)
    
    thickness_list = []
    friction_list = []
    stiffness_list = []
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        thickness_list.append(property_params['thickness'])
        friction_list.append(property_params['dynamic_friction'])
        stiffness_list.append(property_params['cluster_spacing'])
    thickness_stat = {'mean': np.mean(thickness_list), 'std': np.std(thickness_list), 'min': np.min(thickness_list), 'max': np.max(thickness_list)}
    friction_stat = {'mean': np.mean(friction_list), 'std': np.std(friction_list), 'min': np.min(friction_list), 'max': np.max(friction_list)}
    stiffness_stat = {'mean': np.mean(stiffness_list), 'std': np.std(stiffness_list), 'min': np.min(stiffness_list), 'max': np.max(stiffness_list)}
    # save to stat.text for thickness, friction, stiffness
    with open(os.path.join(out_dir, 'stat.txt'), 'w') as f:
        f.write('thickness:')
        f.write(str(thickness_stat))
        f.write('\n')
        f.write('friction:')
        f.write(str(friction_stat))
        f.write('\n')
        f.write('stiffness:')
        f.write(str(stiffness_stat))
    
    # thickness
    plt.figure()
    plt.hist(thickness_list, bins=np.arange(2.5, 4.1, 0.2), edgecolor='black')
    plt.xlabel('thickness')
    plt.ylabel('frequency')
    plt.title('Rope thickness')
    plt.savefig(os.path.join(out_dir, 'thickness.png'))
    plt.close()
    
    # friction
    plt.figure()
    plt.hist(friction_list, bins=np.arange(0.1, 0.5, 0.05), edgecolor='black')
    plt.xlabel('friction')
    plt.ylabel('frequency')
    plt.title('Rope friction')
    plt.savefig(os.path.join(out_dir, 'friction.png'))
    plt.close()
    
    # stiffness
    plt.figure()
    plt.hist(stiffness_list, bins=np.arange(2.0, 8.1, 1.0), edgecolor='black')
    plt.xlabel('stiffness')
    plt.ylabel('frequency')
    plt.title('Rope stiffness')
    plt.savefig(os.path.join(out_dir, 'stiffness.png'))
    plt.close()

def get_rope_epi(data_dir, epi_start, epi_end):
    thickness_list = []
    friction_list = []
    stiffness_list = []
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        thickness_list.append(property_params['thickness'])
        friction_list.append(property_params['dynamic_friction'])
        stiffness_list.append(property_params['cluster_spacing'])
    
    # obtain the epi idx which has the min/max thickness/friction/stiffness
    thickness_min_idx, thickness_max_idx = np.argmin(thickness_list), np.argmax(thickness_list)
    friction_min_idx, friction_max_idx = np.argmin(friction_list), np.argmax(friction_list)
    stiffness_min_idx, stiffness_max_idx = np.argmin(stiffness_list), np.argmax(stiffness_list)
    print(f'thickness min: {thickness_min_idx}, max: {thickness_max_idx}')
    print(f'friction min: {friction_min_idx}, max: {friction_max_idx}')
    print(f'stiffness min: {stiffness_min_idx}, max: {stiffness_max_idx}')

if __name__ == "__main__":
    data_name = 'rope'
    data_dir = f'/mnt/sda/data/{data_name}'
    
    epi_start = 0
    epi_end = 1000
    # get_rope_property_params(data_dir, epi_start, epi_end)
    get_rope_normalized_property_params(data_dir, epi_start, epi_end)
    
    # epi_idx = 999
    # get_eef_pos(data_dir, epi_idx)
    # get_steps(data_dir, epi_idx)
    
    # epi_start = 0
    # epi_end = 1000
    # out_dir = f'/mnt/sda/data_stat/{data_name}'
    # # get_rope_property_stat(data_dir, out_dir, epi_start, epi_end)
    # get_rope_epi(data_dir, epi_start, epi_end)