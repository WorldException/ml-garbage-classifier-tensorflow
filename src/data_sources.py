class DataSource:
    def get_image_shape(self):
        raise NotImplementedError("Pleast implement this method")

    def get_batch(self, batch_size):
        raise NotImplementedError("Pleast implement this method")

    def get_image_shape(self):
        raise NotImplementedError("Pleast implement this method")

class CIFARDataSource(DataSource):
    def __init__(self):
        import cPickle
        import os
        from os.path import join
        import numpy as np
        files = [
            join(os.path.dirname(os.path.abspath(__file__)), 'CIFAR-10_data/cifar-10-batches-py/data_batch_1'),
            join(os.path.dirname(os.path.abspath(__file__)), 'CIFAR-10_data/cifar-10-batches-py/data_batch_2'),
            join(os.path.dirname(os.path.abspath(__file__)), 'CIFAR-10_data/cifar-10-batches-py/data_batch_3'),
            join(os.path.dirname(os.path.abspath(__file__)), 'CIFAR-10_data/cifar-10-batches-py/data_batch_4'),
            join(os.path.dirname(os.path.abspath(__file__)), 'CIFAR-10_data/cifar-10-batches-py/data_batch_5')
        ]
        self.data = None
        self.labels = None

        for file in files:
            with open(file, 'rb') as batch:
                dict = cPickle.load(batch)
                """ data --- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
                The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 blue.
                The image is stored in row-major order, so that the first 32 entries of the array are the red channel
                values of the first row of the image."""
                if self.data is None:
                    data = dict['data'].reshape([-1, 3, 32, 32])
                    self.data = data.transpose([0, 2, 3, 1])
                else:
                    data = dict['data'].reshape([-1, 3, 32, 32])
                    data = data.transpose([0, 2, 3, 1])
                    self.data = np.concatenate((self.data, data), axis=0)

                """ Use one-hot encoding for the labels """
                if self.labels is None:
                    m = np.array(dict['labels']).shape[0]
                    n = 10
                    self.labels = np.zeros((m, n))
                    for index, label in enumerate(dict['labels']):
                        self.labels[index][label] = 1

                else:
                    m = np.array(dict['labels']).shape[0]
                    n = 10
                    labels = np.zeros((m, n))
                    for index, label in enumerate(dict['labels']):
                        labels[index][label] = 1

                    self.labels = np.concatenate((self.labels, labels), axis=0)

    def get_image_shape(self):
        return 32, 32, 3

    def get_batch(self, batch_size=10):
        import random
        m = self.data.shape[0]
        assert batch_size < m
        random_indices = random.sample(range(0, m), batch_size)
        batch_data = []
        batch_labels = []
        for index in random_indices:
            batch_data.append(self.data[index])
            batch_labels.append(self.labels[index])

        return batch_data, batch_labels
