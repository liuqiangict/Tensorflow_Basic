
import tensorflow as tf

a = tf.constant(3)
b = tf.constant(4)

with tf.Session() as sess:
    print("Add: ", sess.run(a + b))
    print("Min: ", sess.run(a - b))
    print("Mul: ", sess.run(a * b))