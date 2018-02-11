import tensorflow as tf
from data_sources import ImageDataSource

"""
Refer to https://www.tensorflow.org/tutorials/deep_cnn
"""

BATCH_SIZE=32
NUM_LABELS=2
ds = ImageDataSource(pathname='../blender-data-generator/renders/labels.json', cached=True)
IMAGE_SHAPE = ds.get_image_shape()

def inference(images):
    """ Conv1 """
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', [5, 5, 3, 64], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    """ Pool1
        Takes and returns a tensor in NHWC format: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    """
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    """ Norm 1
        Do this to prevent saturation: https://www.tensorflow.org/api_guides/python/nn#Normalization
    """
    norm1 = tf.nn.local_response_normalization(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    """ Conv2 """
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', [5, 5, 64, 64], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    """ Norm 2 """
    norm2 = tf.nn.local_response_normalization(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    """ Pool 2 """
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    """ FC1 """
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        weights = tf.get_variable('weights', [IMAGE_SHAPE[0]/4*IMAGE_SHAPE[1]/4*64, 384], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    """ FC2 """
    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable('weights', [384, 192], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

    """ Linear """
    with tf.variable_scope('linear') as scope:
        weights = tf.get_variable('weights', [192, NUM_LABELS], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return linear

def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')

    labels = tf.cast(labels, tf.float32) # Need to do this because labels is tf.int32 while logits is tf.float32
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='sigmoid_cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean

def train():
    images = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
    labels = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])
    with tf.Session() as sess:
        loss_op = loss(inference(images), labels)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

        sess.run(tf.global_variables_initializer())
        for step in range(10000):
            batch = ds.get_batch(BATCH_SIZE)
            batch_images = [item['image'] for item in batch]
            batch_labels = [item['visibility'] for item in batch]

            if step % 100 == 0:
                print 'Loss at step {}: {}'.format(step, loss_op.eval(
                    feed_dict={
                        images: batch_images,
                        labels: batch_labels,
                    }
                ))

            train_step.run(feed_dict={
                images: batch_images,
                labels: batch_labels,
            })

if __name__ == '__main__':
    train()