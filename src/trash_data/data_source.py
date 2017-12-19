import os
from os.path import join
from PIL import Image

def get_data():
    path = '{}/{}'.format(os.path.dirname(os.path.abspath(__file__)), 'dataset/')
    labeled_files = []

    labels = os.listdir(path)
    for label in labels:
        files = os.listdir(join(path, label))
        for file in files:
            full_image_path = join(join(path, label), file)
            labeled_files.append({
                'image': full_image_path,
                'label': label
            })

    print labeled_files[0]