import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from transform_video import merge_video

def get_steps(data_dir, epi_idx):
    steps_path = os.path.join(data_dir, f"episode_{epi_idx}/steps.npy")
    steps = np.load(steps_path)
    print(f"Episode {epi_idx} steps: {steps}")

def get_eef_pos(data_dir, epi_idx):
    eef_pos_path = os.path.join(data_dir, f"episode_{epi_idx}/eef_pos.npy")
    eef_pos = np.load(eef_pos_path)
    print(f"Episode {epi_idx} eef pos: {eef_pos.shape}")

"""
Rope
"""
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
    stiffness_list = []
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        stiffness_list.append(property_params['stiffness'])
    
    # obtain the epi idx which has the min/max thickness/friction/stiffness
    stiffness_min_idx, stiffness_max_idx = np.argmin(stiffness_list), np.argmax(stiffness_list)
    print(f'stiffness min: {stiffness_min_idx} with stiffness: {stiffness_list[stiffness_min_idx]}')
    print(f'stiffness max: {stiffness_max_idx} with stiffness: {stiffness_list[stiffness_max_idx]}')
    
    # obtain the epi idx which has average stiffness
    stiffness_list = np.array(stiffness_list)
    avg_stiffness = np.mean(stiffness_list)
    avg_stiffness_idx = np.argmin(np.abs(stiffness_list - avg_stiffness))
    print(f'avg stiffness idx: {avg_stiffness_idx} with stiffness: {stiffness_list[avg_stiffness_idx]}')

def get_rope_mpp_epi(data_dir, epi_start, epi_end, plot=False):
    stiffness_list, friction_list = [], [] 
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        stiffness_list.append(property_params['stiffness'])
        friction_list.append(property_params['dynamic_friction'])
    
    stiffness_list, friction_list = np.array(stiffness_list), np.array(friction_list)
    if plot:
        # draw a heatmap
        data_stat_path = '/mnt/sda/adaptigraph/data_stat'
        os.makedirs(data_stat_path, exist_ok=True)
        
        ### plot
        ## add more layers onto the plot
        # g = sns.jointplot(x=stiffness_list, y=friction_list)
        # g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
        # g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
        
        ## hist plot
        sns.jointplot(x=stiffness_list, y=friction_list, kind='hex')
        
        plt.size = (40, 40)
        plt.xlabel('stiffness')
        plt.ylabel('friction')
        plt.savefig(os.path.join(data_stat_path, 'rope_stiffness_friction.png'))
    
    # obtain the epi idx which has the friction > 0.8 and stiffness < 0.1
    # idx = np.where((friction_list > 0.6) & (friction_list < 0.7) & (stiffness_list < 0.1))[0]
    idx = np.where((friction_list > 0.17) & (friction_list < 0.18) & (stiffness_list < 0.1))[0]
    print(idx)
    
    for i in idx[:5]:
        print(f'episode {i}, stiffness: {stiffness_list[i]}, friction: {friction_list[i]}')
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        video_dir = f'/mnt/sda/adaptigraph/data_viz/rope'
        video_path = os.path.join(video_dir, f'episode_{i}.mp4')
        os.makedirs(video_dir, exist_ok=True)
        merge_video(os.path.join(epi_dir, 'camera_0'), video_path)
    
"""
Cloth
"""
def get_cloth_property_params(data_dir, epi_start, epi_end):
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        print(f'Episode {i}, sf: {property_params["sf"]} , stiffness: {property_params["bend_stiffness"]}, friction: {property_params["dynamic_friction"]}')

def get_cloth_epi(data_dir, epi_start, epi_end):
    sf_list = []
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        sf_list.append(property_params['sf'])
    
    # obtain the epi idx which has the min/max thickness/friction/stiffness
    sf_min_idx, sf_max_idx = np.argmin(sf_list), np.argmax(sf_list)
    print(f'sf min: {sf_min_idx} with sf: {sf_list[sf_min_idx]}')
    print(f'sf max: {sf_max_idx} with sf: {sf_list[sf_max_idx]}')
    
    # obtain the epi idx which has average stiffness
    sf_list = np.array(sf_list)
    avg_sf = np.mean(sf_list)
    avg_sf_idx = np.argmin(np.abs(sf_list - avg_sf))
    print(f'avg sf idx: {avg_sf_idx} with sf: {sf_list[avg_sf_idx]}')
    
    # obtain the epi idx which has sf values
    sf_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for sf in sf_values:
        sf_idx = np.argmin(np.abs(sf_list - sf))
        print(f'sf {sf} idx: {sf_idx} with sf: {sf_list[sf_idx]}')

def get_cloth_property_stat(data_dir, out_dir, epi_start, epi_end):
    
    os.makedirs(out_dir, exist_ok=True)
    
    sf_list = []
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        sf_list.append(property_params['sf'])
    sf_stat = {'mean': np.mean(sf_list), 'std': np.std(sf_list), 'min': np.min(sf_list), 'max': np.max(sf_list)}
    # save to stat.text for thickness, friction, stiffness
    with open(os.path.join(out_dir, 'stat.txt'), 'w') as f:
        f.write('sf:')
        f.write(str(sf_stat))
    
    # df
    plt.figure()
    plt.hist(sf_list, bins=np.arange(0.0, 1.1, 0.1), edgecolor='black')
    plt.xlabel('sf')
    plt.ylabel('frequency')
    plt.title('Cloth sf')
    plt.savefig(os.path.join(out_dir, 'sf.png'))

"""
Granular
"""
def get_granular_epi(data_dir, epi_start, epi_end):
    scale_list = []
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        scale_list.append(property_params['granular_scale'])
    
    # obtain the epi idx which has the min/max thickness/friction/stiffness
    scale_min_idx, scale_max_idx = np.argmin(scale_list), np.argmax(scale_list)
    print(f'scale min: {scale_min_idx} with scale: {scale_list[scale_min_idx]}')
    print(f'scale max: {scale_max_idx} with scale: {scale_list[scale_max_idx]}')
    
    # obtain the epi idx which has average stiffness
    scale_list = np.array(scale_list)
    avg_scale = np.mean(scale_list)
    avg_scale_idx = np.argmin(np.abs(scale_list - avg_scale))
    print(f'avg scale idx: {avg_scale_idx} with scale: {scale_list[avg_scale_idx]}')
    
    # obtain the epi idx which has sf values
    scale_values = [0.1, 0.2, 0.3]
    for scale in scale_values:
        scale_idx = np.argmin(np.abs(scale_list - scale))
        print(f'scale {scale} idx: {scale_idx} with scale: {scale_list[scale_idx]}')
        
def get_granular_mpp_epi(data_dir, epi_start, epi_end, plot=False):
    scale_list, friction_list = [], [] 
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        scale_list.append(property_params['granular_scale'])
        friction_list.append(property_params['dynamic_friction'])
    
    scale_list, friction_list = np.array(scale_list), np.array(friction_list)
    if plot:
        # draw a heatmap
        data_stat_path = '/mnt/sda/adaptigraph/data_stat'
        os.makedirs(data_stat_path, exist_ok=True)
        
        ### plot
        ## add more layers onto the plot
        # g = sns.jointplot(x=scale_list, y=friction_list)
        # g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
        # g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
        
        ## hist plot
        sns.jointplot(x=scale_list, y=friction_list, kind='hex')
        
        plt.size = (40, 40)
        plt.xlabel('scale')
        plt.ylabel('friction')
        plt.savefig(os.path.join(data_stat_path, 'granular_scale_friction.png'))
    
    # obtain the epi idx which has the friction > 0.8 and scale < 0.1
    idx = np.where((friction_list > 0.9) & (friction_list < 1.0) & (scale_list < 0.2))[0]
    print(idx)
    
    for i in idx[:5]:
        print(f'episode {i}, scale: {scale_list[i]}, friction: {friction_list[i]}')
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        video_dir = f'/mnt/sda/adaptigraph/data_viz/granular'
        video_path = os.path.join(video_dir, f'episode_{i}.mp4')
        os.makedirs(video_dir, exist_ok=True)
        merge_video(os.path.join(epi_dir, 'camera_0'), video_path)


if __name__ == "__main__":
    
    epi_start = 0
    epi_end = 1000
    
    data_name = 'rope_0402'
    data_dir = f'/mnt/sda/data/{data_name}'
    get_rope_mpp_epi(data_dir, epi_start, epi_end)
    
    # data_dir = f'/mnt/sda/data/granular/carrots'
    # get_granular_mpp_epi(data_dir, epi_start, epi_end, plot=True)
        
        