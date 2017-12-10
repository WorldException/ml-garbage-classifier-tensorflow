import tensorflow as tf
import numpy as np

np.random.seed(0)
x_value = np.random.rand(2, 3)
x = tf.constant(x_value, dtype=tf.float32, name="x")
y = tf.placeholder(dtype=tf.float32, shape=[3, 1], name="y")

matmul = tf.matmul(x, y)

with tf.Session() as sess:
    y_value = np.random.rand(3, 1)
    tf_computed = sess.run(matmul, feed_dict={y: y_value})

print "TF Computed:"
print tf_computed
print "Expected:"
print np.dot(x_value, y_value)