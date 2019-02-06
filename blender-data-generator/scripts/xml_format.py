#!/usr/bin/env python
#coding:utf8
from __future__ import unicode_literals, print_function
import os

tmpl_body="""<annotation>
    <folder>{folder}</folder>
    <filename>{filename}</filename>
    <path>{fullpath}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    {labels}
</annotation>
"""

tmpl_object="""
    <object>
        <name>{name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{box[x1]}</xmin>
            <ymin>{box[y1]}</ymin>
            <xmax>{box[x2]}</xmax>
            <ymax>{box[y2]}</ymax>
        </bndbox>
    </object>
"""


def dump_labels(labels):
    xml_labels = ''
    for name, box in labels['meshes'].items():
        xml_labels += tmpl_object.format(
            name=name,
            box=box
        )
    filename = labels['fullpath']
    name, ext = os.path.splitext(filename)
    filexml = name + '.xml'

    xml=tmpl_body.format(
        folder='',
        filename=filename,
        fullpath=filename,
        width=0,
        height=0,
        labels=xml_labels
    )

    print(filexml)
    with open(filexml, 'w') as f:
        f.write(xml)


if __name__ == '__main__':
    test = {
        'image': 'test.png',
        'fullpath': 'test.png',
        'meshes': {
            'box1': {
                        'x1': 0,
                        'y1': 1,
                        'x2': 100,
                        'y2': 200
                    }
        }
    }
    dump_labels(test)