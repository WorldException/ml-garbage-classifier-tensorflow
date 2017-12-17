from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import numpy as np
import matplotlib.pyplot as plt
import time

imgplot = None
plt.ion()

for i in range(0, 10):
    image = mnist.train.images[i]
    image = np.reshape(image, (28, 28))

    if imgplot is None:
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots
        fig, ax = plt.subplots()
        ax.set(title='MNIST Data')
        imgplot = ax.imshow(image)

    imgplot.set_data(image)
    ax.text(1, 1, mnist.train.labels[i], bbox=dict(facecolor='white'))

    fig.canvas.draw()
    time.sleep(1)