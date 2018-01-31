import bpy, os
from math import sin, cos, pi
import numpy as np
import boundingbox
import json

camera = bpy.data.objects['Camera']
radians_in_circle = 2.0 * pi
steps = 10

original_position = np.matrix([
    [8],
    [0],
    [2]
])

""" This will store the bonding boxes """
labels = []

for i in range(0, steps + 1):
    for j in range(0, steps + 1):
        yaw = radians_in_circle * (i / steps)
        pitch = -1.0 * radians_in_circle / 16.0 * (j / steps)
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
        camera.location.x = new_position[0][0]
        camera.location.y = new_position[1][0]
        camera.location.z = new_position[2][0]
        
        # Rendering
        # https://blender.stackexchange.com/questions/1101/blender-rendering-automation-build-script
        filename = '{}y-{}p.png'.format(str(i), str(j))
        bpy.context.scene.render.filepath = os.path.join('./renders/', filename)
        bpy.ops.render.render(write_still=True)

        """ Get the bounding box coordinates """
        scene = bpy.data.scenes['Scene']
        cube = bpy.data.objects['Cube']
        bounding_box = boundingbox.camera_view_bounds_2d(scene, camera, cube)
        label_entry = {
            'image': filename,
            'name': cube.name
        }
        if boundingbox:
            label_entry['bounding_box'] = {
                'x1': bounding_box[0][0],
                'y1': bounding_box[0][1],
                'x2': bounding_box[1][0],
                'y2': bounding_box[1][1]
            }
        else:
            label_entry['bounding_box'] = None

        labels.append(label_entry)

    """ Write labels to file """
    with open('./renders/labels.json', 'w+') as f:
        json.dump(labels, f, sort_keys=True, indent=4, separators=(',', ': '))