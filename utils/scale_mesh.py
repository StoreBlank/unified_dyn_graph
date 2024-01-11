import numpy as np
import trimesh

def create_cube_mesh(width, height, depth):
    vertices = np.array([
        [0, 0, 0],                # Vertex 0
        [width, 0, 0],            # Vertex 1
        [width, height, 0],       # Vertex 2
        [0, height, 0],           # Vertex 3
        [0, 0, depth],            # Vertex 4
        [width, 0, depth],        # Vertex 5
        [width, height, depth],   # Vertex 6
        [0, height, depth]        # Vertex 7
    ])

    # Each face is made of 2 triangles
    triangles = np.array([
        [0, 3, 1], [1, 3, 2],     # Front face
        [1, 2, 5], [2, 6, 5],     # Right face
        [5, 6, 4], [6, 7, 4],     # Back face
        [4, 7, 0], [7, 3, 0],     # Left face
        [3, 7, 2], [7, 6, 2],     # Top face
        [0, 1, 4], [1, 5, 4]      # Bottom face
    ])

    return vertices, triangles

width, height, depth = 20, 5, 10  # Cube dimensions
vertices, triangles = create_cube_mesh(width, height, depth)
mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

from io import StringIO

# Function to create the PLY file content
def create_ply_content(vertices, faces):
    ply_content = StringIO()
    ply_content.write("ply\n")
    ply_content.write("format ascii 1.0\n")
    ply_content.write(f"element vertex {len(vertices)}\n")
    ply_content.write("property float x\n")
    ply_content.write("property float y\n")
    ply_content.write("property float z\n")
    ply_content.write(f"element face {len(faces)}\n")
    ply_content.write("property list uchar int vertex_indices\n")
    ply_content.write("end_header\n")

    # Write vertices
    for v in vertices:
        ply_content.write(f"{v[0]} {v[1]} {v[2]}\n")

    # Write faces
    for f in faces:
        ply_content.write(f"3 {f[0]} {f[1]} {f[2]}\n")

    return ply_content.getvalue()

# Create the PLY file content
ply_content = create_ply_content(vertices, triangles)

# Save to a file
ply_filename = '/home/baoyu/2023/unified_dyn_graph/assets/mesh/cube_mesh.ply'
with open(ply_filename, 'w') as file:
    file.write(ply_content)





# import numpy as np
# from plyfile import PlyData, PlyElement

# def scale_ply_cube(input_file, output_file, scale_factors):
#     # Load the PLY file
#     plydata = PlyData.read(input_file)

#     # Extract the vertex data
#     vertex = plydata['vertex']
#     x = vertex['x']
#     y = vertex['y']
#     z = vertex['z']

#     # Apply scaling
#     x *= scale_factors[0]
#     y *= scale_factors[1]
#     z *= scale_factors[2]

#     # Update the vertex data
#     vertex['x'] = x
#     vertex['y'] = y
#     vertex['z'] = z

#     # Write the scaled data to a new PLY file
#     PlyData([vertex, plydata['face']], text=True).write(output_file)

# # Example usage
# input_ply_file = '/home/hanxiao/Desktop/Mingtong/box.ply'
# output_ply_file = '/home/hanxiao/Desktop/Mingtong/box_scaled.ply'
# scale_factors = (3, 2, 4)  # Scale factors for width, length, and depth

# scale_ply_cube(input_ply_file, output_ply_file, scale_factors)