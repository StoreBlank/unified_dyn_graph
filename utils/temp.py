import os
import open3d as o3d

tool_mesh_path = '/home/baoyu/2023/unified_dyn_graph/assets/mesh/dustpan.obj'
tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
tool_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(tool_mesh, 50)
o3d.visualization.draw_geometries([tool_surface, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)])