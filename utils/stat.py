import os
import numpy as np
import json
import argparse

def get_property_params(data_dir, epi_start, epi_end):
    for i in range(epi_start, epi_end):
        epi_dir = os.path.join(data_dir, f'episode_{i}')
        with open(os.path.join(epi_dir, 'property_params.json'), 'r') as f:
            property_params = json.load(f)
        print(f'Episode {i}, thickness: {property_params["thickness"]}, friction: {property_params["dynamic_friction"]}, stiffness: {property_params["cluster_spacing"]}')

if __name__ == "__main__":
    data_name = 'rope'
    data_dir = f'/mnt/sda/data/{data_name}'
    epi_start = 0
    epi_end = 10
    get_property_params(data_dir, epi_start, epi_end)