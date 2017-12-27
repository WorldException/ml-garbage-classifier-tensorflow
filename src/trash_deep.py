import tensorflow as tf
from trash_data.data_source import DataSource

ds = DataSource()
ti, tl, vi, vl = ds.get_data()
n_classes = tl[0].shape[0]

x = tf.placeholder(tf.float32, shape=[None, 512 * 384])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

""" Weight initializers """
def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Convolution layers helpers """
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

""" Model setup -------------- """

""" Conv Layer 1 """
""" 32 features per 7x7 patch (patch size, patch size, n input channels, n output channels)"""
W_conv1 = weight_variables([7, 7, 1, 16])
b_conv1 = bias_variable([16])
""" Reshape x to a 4d tensor """
""" inferred (-1), width, height, n channels"""
x_image = tf.reshape(x, [-1, 512, 384, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
""" Pooling
    Takes and returns a tensor in NHWC format: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    (n samples, height, width, channels)
"""
h_pool1 = max_pool_2x2(h_conv1)

""" Conv Layer 2 """
W_conv2 = weight_variables([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

""" FC Layer 1 """
""" /4 since the input images goes through 2 pooling layers """
W_fc1 = weight_variables([512/4 * 384/4 * 32, 10])
b_fc1 = bias_variable([10])
h_pool2_flat = tf.reshape(h_pool2, [-1, 512/4 * 384/4 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

""" Dropout Layer """
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

""" FC Layer 2 """
W_fc2 = weight_variables([10, n_classes])
b_fc2 = bias_variable([n_classes])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2


""" Train it """
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


""" Prep for plotting """
from conv_weight_plot import WeightPlotter
weight_plotter = WeightPlotter(4, 4)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20000):
        batch_images, batch_labels = ds.get_random_train_batch(50)

        if step % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_images,
                y_: batch_labels,
                keep_prob: 1.0
            })
            print('step %d, training accuracy %g' % (step, train_accuracy))

            """ Plot the weights for Conv1 """
            w = W_conv1.eval()
            weight_plotter.plot(w)

        train_step.run(feed_dict={
            x: batch_images,
            y_: batch_labels,
            keep_prob: 0.6
        })

# print('test accuracy %g' % accuracy.eval(feed_dict={
#     x: mnist.test.images,
#     y_: mnist.test.labels,
#     keep_prob: 1.0
# }))