import bpy, os
from math import sin, cos, pi
import numpy as np
import json
import sys

"""
Add scripts folder to Blender's Python interpreter and reload all scripts.
http://web.purplefrog.com/~thoth/blender/python-cookbook/import-python.html
"""
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

import xml_format
import boundingbox
import importlib
importlib.reload(boundingbox)

render_dir = './render'

def render(scene, camera_object, mesh_objects, camera_steps, file_prefix="render"):
    """
    Renders the scene at different camera angles to a file, and returns a list of label data
    """

    radians_in_circle = 2.0 * pi
    original_position = np.matrix([
        [8],
        [0],
        [2]
    ])

    """ This will store the bonding boxes """
    labels = []

    for i in range(0, camera_steps + 1):
        for j in range(0, camera_steps + 1):
            yaw = radians_in_circle * (i / camera_steps)
            pitch = -1.0 * radians_in_circle / 16.0 * (j / camera_steps)
            # Blender uses a Z-up coordinate system instead of the standard Y-up system, therefor:
            # yaw = rotate around z-axis
            # pitch = rotate around y-axis
            yaw_rotation_matrix = np.matrix([
                [cos(yaw), -sin(yaw), 0],
                [sin(yaw), cos(yaw), 0],
                [0, 0, 1]
            ])
            pitch_rotation_matrix = np.matrix([
                [cos(pitch), 0, sin(pitch)],
                [0, 1, 0],
                [-sin(pitch), 0, cos(pitch)]
            ])

            new_position = yaw_rotation_matrix * pitch_rotation_matrix * original_position
            camera_object.location.x = new_position[0][0]
            camera_object.location.y = new_position[1][0]
            camera_object.location.z = new_position[2][0]

            # Rendering
            # https://blender.stackexchange.com/questions/1101/blender-rendering-automation-build-script
            filename = '{}-{}y-{}p.png'.format(str(file_prefix), str(i), str(j))
            bpy.context.scene.render.filepath = os.path.join(render_dir, filename)
            bpy.ops.render.render(write_still=True)

            scene = bpy.data.scenes['Scene']
            w, h = scene.render.resolution_x, scene.render.resolution_y
            label_entry = {
                'image': filename,
                'fullpath': os.path.join(render_dir, filename),
                'width': w,
                'height': h,
                'meshes': {}
            }

            """ Get the bounding box coordinates for each mesh """
            for object in mesh_objects:
                bounding_box = boundingbox.camera_view_bounds_2d(scene, camera_object, object)
                if bounding_box:
                    """
                    пересчет с учетом того начала системы координат в левом верхнем углу                    
                    """

                    label_entry['meshes'][object.name] = {
                        'x1': bounding_box[0][0],
                        'y1': bounding_box[0][1],
                        'x2': bounding_box[1][0],
                        'y2': bounding_box[1][1],

                        'px1': round(bounding_box[0][0] * w),
                        'py1': h - round(bounding_box[1][1] * h),
                        'px2': round(bounding_box[1][0] * w),
                        'py2': h - round(bounding_box[0][1] * h)
                    }

            labels.append(label_entry)
            # export xml boxes
            xml_format.dump_labels(label_entry)
        break
    return labels


def batch_render(scene, camera_object, mesh_objects):
    import scene_setup
    camera_steps = 10
    scene_setup_steps = 1
    spawn_range = [
        (-10, 10),
        (-10, 10),
        (5, 10)
    ]
    labels = []

    for i in range(0, scene_setup_steps):
        scene_setup.simulate(scene, mesh_objects, spawn_range, 0.75)
        scene_labels = render(scene, camera_object, mesh_objects, camera_steps, file_prefix=i)
        labels += scene_labels # Merge lists

    with open(os.path.join(render_dir, 'labels.json'), 'w+') as f:
        json.dump(labels, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    scene = bpy.data.scenes['Scene']
    camera_object = bpy.data.objects['Camera']
    mesh_names = ['Cube', 'Sphere']
    mesh_objects = [bpy.data.objects[name] for name in mesh_names]
    batch_render(scene, camera_object, mesh_objects)