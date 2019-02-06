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
            <xmin>{box[px1]:.0f}</xmin>
            <ymin>{box[py1]:.0f}</ymin>
            <xmax>{box[px2]:.0f}</xmax>
            <ymax>{box[py2]:.0f}</ymax>
            
            <axmin>{box[x1]}</axmin>
            <aymin>{box[y1]}</aymin>
            <axmax>{box[x2]}</axmax>
            <aymax>{box[y2]}</aymax>
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
        width=labels['width'],
        height=labels['height'],
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