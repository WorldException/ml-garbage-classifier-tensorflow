class DataSource:
    def get_image_shape(self):
        raise NotImplementedError("Pleast implement this method")

    def get_batch(self, batch_size):
        """

        :return: features, labels
        """
        raise NotImplementedError("Pleast implement this method")

    def get_image_shape(self):
        """

        :return: image width, image height, image depth
        """
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


class ImageDataSource(DataSource):
    """
    Expects a directory of subdirectories, with each subdirectory being a label and containing images.
    """
    def __init__(self, path, shape=None, ignored_labels=[]):
        import os
        from os.path import join
        import numpy as np
        from PIL import Image

        self.data = None
        self.labels = None

        labeled_images = []
        subdirs = os.listdir(path)
        n_labels = len(subdirs)

        """ Retrieve all the image pathnames and labels """
        for index, label in enumerate(subdirs):
            """ Skip ignored lables """
            if label in ignored_labels:
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
            image = Image.open(labeled_image['image_path']).convert('RGB')

            if shape is not None:
                image = image.resize(shape, resample=Image.BILINEAR)

            """
            Fast PIL > numpy conversion
            https://stackoverflow.com/questions/13550376/pil-image-to-array-numpy-array-to-array-python/42036542#42036542
            """
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = np.reshape(im_arr, (image.size[0], image.size[1], 3))
            labeled_image['array'] = im_arr

        self.data = labeled_images

    def get_image_shape(self):
        return self.data[0]['array'].shape[0], self.data[0]['array'].shape[1], 3

    def get_batch(self, batch_size=10):
        assert batch_size <= len(self.data)
        import random
        shuffled_data = self.data[:]
        random.shuffle(shuffled_data)
        batch = shuffled_data[0:batch_size]
        return [image['array'] for image in batch],\
               [image['label'] for image in batch]


class FlatImageDataSource(DataSource):
    def __init__(self, pathname, cached=False):
        import os
        import json
        import pickle
        import numpy
        from PIL import Image

        dir_name = os.path.dirname(pathname)
        pickle_pathname = os.path.join(dir_name, 'labels.pickle')
        if cached and os.path.isfile(pickle_pathname):
            with open(pickle_pathname, 'rb') as fp:
                self.data = pickle.load(fp)

        else:
            with open(pathname) as fp:
                data = json.load(fp)
                fp.close()

            self.data = []
            visibility_matrices = self.__get_visibility_matrices(data)

            for i, entry in enumerate(data):
                image_path = os.path.abspath(os.path.join(dir_name, entry['image']))
                image = Image.open(image_path).convert('RGB')
                image_array = numpy.fromstring(image.tobytes(), dtype=numpy.uint8)
                image_array = numpy.reshape(image_array, (image.size[0], image.size[1], 3))
                self.data.append({
                    'image': image_array,
                    'visibility': visibility_matrices[i],
                    'bounding_boxes': [value for key, value in entry['meshes'].items()]
                })

            with open(pickle_pathname, 'wb+') as fp:
                pickle.dump(self.data, fp)

        print self.data

    def get_image_shape(self):
        return self.data[0]['image'].shape[0], self.data[0]['image'].shape[1], 3

    def get_batch(self, batch_size=10):
        assert batch_size <= len(self.data)
        import random
        shuffled_data = self.data[:]
        random.shuffle(shuffled_data)
        batch = shuffled_data[0:batch_size]
        return batch

    def __get_visibility_matrices(self, data):
        """
        :param data: The contents of the label file
        :type data: dict
        :return: A list of one-hot vectors
        """
        import numpy

        """ Figure out how many unique names there are """
        names = set()
        for entry in data:
            for mesh in entry['meshes']:
                names.add(mesh)

        n_unique_names = len(names)

        matrices = []
        for entry in data:
            matrix = numpy.zeros(shape=(n_unique_names, 1))
            for i, name in enumerate(entry['meshes']):
                matrix[i, 0] = 1.

            matrices.append(matrix)

        return matrices
