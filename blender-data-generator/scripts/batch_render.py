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
import sys
import datetime
import importlib
importlib.reload(boundingbox)

u"""
Настройки
"""
# автоматическое определение пути в зависимости от платформы
if sys.platform == 'linux':
    render_dir = './render'
else:
    render_dir = 'c:\\render-%s' % datetime.datetime.now().strftime('%Y-%m-%d_%H%M')

# сколько раз повернуть
camera_steeps = 5

# исходные позиции камеры
cam_positions = """
8, 0, 4
-8, 0, 4
0, -8, 4
0, 8, 4
6,93, 4, 4
4, 6,93, 4
6,93, -4, 4
-6,93, 4, 4
-6,93, -4, 4
4, -6,93, 4
-4, 6,93, 4
-4, -6,93, 4
"""
# переобразую текст в массив
cams = np.matrix(cam_positions.replace(', ', ' ').replace(',', '.'))
print('shape', cams.shape)
#n.reshape(12,3,1)[0].reshape(3,1)
# колво чисел в массиве
cams_len = cams.shape[1]
cams_positions = int(cams_len / 3);
print('cams position', cams_positions)
# получаю
cams = cams.reshape(cams_positions, 3)
# позиция
#cams[0].reshape(3)


def save_render(file_prefix, i, j, mesh_objects):
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

    # export xml boxes
    xml_format.dump_labels(label_entry)
    return label_entry


def render_cams(cams, scene, camera_object, mesh_objects, file_prefix="render_cam"):
    labels = []
    for i, cam in enumerate(cams):
        print(cam)
        new_position = cam.reshape(3, 1)
        camera_object.location.x = new_position[0]
        camera_object.location.y = new_position[1]
        camera_object.location.z = new_position[2]
        labels.append(save_render(file_prefix, i, 0, mesh_objects))

    return labels


def render_cams_steeps(cams, scene, camera_object, mesh_objects, camera_steeps, file_prefix="render_cam"):
    labels = []
    for i, cam in enumerate(cams):
        print(cam)
        new_position = cam.reshape(3, 1)
        render(scene, camera_object, mesh_objects, camera_steeps, '%s-%s' % (file_prefix, i), start_position=new_position)


def render(scene, camera_object, mesh_objects, camera_steps, file_prefix="render", start_position=None):
    """
    Renders the scene at different camera angles to a file, and returns a list of label data
    """

    radians_in_circle = 2.0 * pi
    if not start_position is None:
        original_position = start_position
    else:
        original_position = np.matrix([
            [8],
            [0],
            [2]
        ])
    print('original', original_position)
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
                [-sin(pitch), 0, cos(pitch)],
                [0, 0, 1],
            ])

            new_position = yaw_rotation_matrix * pitch_rotation_matrix * original_position
            camera_object.location.x = new_position[0][0]
            camera_object.location.y = new_position[1][0]
            camera_object.location.z = new_position[2][0]

            labels.append(save_render(file_prefix, i, j, mesh_objects))

    return labels


if __name__ == '__main__':
    scene = bpy.data.scenes['Scene']
    camera_object = bpy.data.objects['Camera']
    mesh_names = ['Cube', 'Sphere']
    mesh_objects = [bpy.data.objects[name] for name in mesh_names]
    render_cams_steeps(cams, scene, camera_object, mesh_objects, camera_steeps=camera_steeps)
