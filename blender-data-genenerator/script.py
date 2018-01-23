import bpy, os
from math import sin, cos, pi
import numpy as np

camera = bpy.data.objects['Camera']
radians_in_circle = 2.0 * pi
steps = 20

original_position = np.matrix([
    [8],
    [0],
    [0]
])

for i in range(0, steps + 1):
    for j in range(0, steps + 1):
        yaw = radians_in_circle * (i / steps)
        pitch = radians_in_circle * (j / steps)
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
        bpy.context.scene.render.filepath = os.path.join('./renders/', '{}y-{}p'.format(str(i), str(j)))
        bpy.ops.render.render(write_still=True)