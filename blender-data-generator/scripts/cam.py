#!/usr/bin/env python
#coding:utf8
from __future__ import unicode_literals, print_function
import bpy, os
from math import sin, cos, pi
import numpy as np

camera_object = bpy.data.objects['Camera']

camera_steps = 1
radians_in_circle = 2.0 * pi
print(camera_object.location)
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
            [-sin(pitch), 0, cos(pitch)],
            [0, 0, 1],
        ])

        new_position = yaw_rotation_matrix * pitch_rotation_matrix * original_position
        print(new_position)
        camera_object.location.x = new_position[0][0]
        camera_object.location.y = new_position[1][0]
        camera_object.location.z = new_position[2][0]