import os
from os import devnull
import subprocess
from pathlib import Path
import imageio
import OpenEXR
from Imath import PixelType

import numpy as np
import pyflex 
import trimesh


def grid_index(x, y, dimx):
    return y*dimx + x

def get_cloth_mesh(dimx, dimy, base_index=0):
    if dimx == -1 or dimy == -1:
        positions = pyflex.get_positions().reshape((-1, 4))
        vertices = positions[:, :3]
        faces = pyflex.get_faces().reshape((-1, 3))
    else:
        positions = pyflex.get_positions().reshape((-1, 4))
        faces = []
        vertices = positions[:, :3]
        for y in range(dimy):
            for x in range(dimx):
                if x > 0 and y > 0:
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y-1, dimx),
                        base_index + grid_index(x, y, dimx)
                    ])
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y, dimx),
                        base_index + grid_index(x-1, y, dimx)])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def blender_render_cloth(cloth_mesh, resolution):
    output_prefix = '/tmp/' + str(os.getpid())
    obj_path = output_prefix + '.obj'
    cloth_mesh.export(obj_path)
    commands = [
        'blender',
        'cloth.blend',
        '-noaudio',
        '-E', 'BLENDER_EEVEE',
        '--background',
        '--python',
        'render_rgbd.py',
        obj_path,
        output_prefix,
        str(resolution)]
    with open(devnull, 'w') as FNULL:
        while True:
            try:
                # render images
                subprocess.check_call(
                    commands,
                    stdout=FNULL)
                break
            except Exception as e:
                print(e)
    # get images
    output_dir = Path(output_prefix)
    color = imageio.imread(str(list(output_dir.glob('*.png'))[0]))
    color = color[:, :, :3]
    depth = OpenEXR.InputFile(str(list(output_dir.glob('*.exr'))[0]))
    redstr = depth.channel('R', PixelType(PixelType.FLOAT))
    depth = np.fromstring(redstr, dtype=np.float32)
    depth = depth.reshape(resolution, resolution)
    return color, depth
