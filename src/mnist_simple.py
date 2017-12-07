""" https://www.tensorflow.org/get_started/mnist/beginners """

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # Downloads MNIST data to src folder

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784]) # m x number of features

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

""" Training --------------"""

""" The model """
y = tf.nn.softmax(logits=tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10]) # The label data

""" This cost function
    Could also use tf.nn.softmax_cross_entropy_with_logits instead.
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # The cost function
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

""" Evaluate -----------------"""

""" Get index of largest predicted value and check that it matches the label
    Ex: [True, False, True, True]
"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
""" tf.cast: [True, False, True, True] -> [1, 0, 1, 1] """
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))