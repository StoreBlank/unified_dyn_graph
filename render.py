import os
import pathlib
import numpy as np

# TODO
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