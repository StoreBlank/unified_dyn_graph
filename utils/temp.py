# import os
# import open3d as o3d

# tool_mesh_path = '/home/baoyu/2023/unified_dyn_graph/assets/mesh/dustpan.obj'
# tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
# tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 50)
# o3d.visualization.draw_geometries([tool_surface, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)])

import numpy as np
import json

data_dir = '/mnt/sda/data/cloth'
epi_idx = np.random.randint(0, 1000)
file = f"{data_dir}/episode_{epi_idx}/property_params.json"
with open(file) as f:
    property_params = json.load(f)
print(property_params['num_particles'])