import os
from os.path import join
from PIL import Image
import numpy as np
import random

class DataSource():
    def __init__(self, percent_train=0.8, shape=None):
        assert percent_train < 1.0
        self.train = None
        self.validate = None
        self.percent_train=percent_train
        self.shape = shape

    def get_data(self):
        if self.train is not None:
            return [image['array'] for image in self.train], \
                   [image['label'] for image in self.train], \
                   [image['array'] for image in self.validate], \
                   [image['label'] for image in self.validate]

        path = '{}/{}'.format(os.path.dirname(os.path.abspath(__file__)), 'dataset/')
        labeled_images = []

        """ Find the image pathnames, where each folder is a label. """
        folders = os.listdir(path)
        n_labels = len(folders)
        ignore_labels = ['cardboard', 'plastic', 'metal', 'trash']

        for index, label in enumerate(folders):
            """ Skip ignored labels """
            if label in ignore_labels:
                continue

            one_hot = np.zeros(n_labels)
            one_hot[index] = 1
            files = os.listdir(join(path, label))
            for file in files:
                full_image_path = join(join(path, label), file)
                labeled_images.append({
                    'image_path': full_image_path,
                    'label': one_hot
                })

        """ Load image data from pathnames """
        for labeled_image in labeled_images:
            """ The added .convert("L") converts the image to greyscale """
            image = Image.open(labeled_image['image_path']).convert("L")

            if self.shape is not None:
                image = image.resize(self.shape, resample=Image.BILINEAR)

            """
            Fast PIL > numpy conversion
            https://stackoverflow.com/questions/13550376/pil-image-to-array-numpy-array-to-array-python/42036542#42036542
            """
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = np.reshape(im_arr, (image.size[0], image.size[1]))
            labeled_image['array'] = im_arr

        """ Assign images to train/validate lists """
        n_train = int(len(labeled_images) * self.percent_train)
        n_validate = len(labeled_images) - n_train
        random.shuffle(labeled_images)
        """ a[start:end] start (inclusive) to end (exclusive)"""
        self.train = labeled_images[0:n_train]
        self.validate = labeled_images[n_train:n_train + n_validate]

        return [image['array'] for image in self.train], \
               [image['label'] for image in self.train], \
               [image['array'] for image in self.validate], \
               [image['label'] for image in self.validate]

    def get_random_train_batch(self, batch_size=10):
        if self.train is None:
            self.get_data()

        assert batch_size <= len(self.train)
        shuffled_train = self.train[:]
        random.shuffle(shuffled_train)
        batch = shuffled_train[0:batch_size]
        return [image['array'].reshape(-1) for image in batch], \
               [image['label'] for image in batch], \

    def get_shape(self):
        return self.get_data()[0][0].shape
