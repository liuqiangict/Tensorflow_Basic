
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.div(a, b)

with tf.Session() as sess:
    print("Add: ", sess.run(add, feed_dict={a : 3, b : 4}))
    print("Sub: ", sess.run(sub, feed_dict={a : 3, b : 4}))
    print("Mul: ", sess.run(mul, feed_dict={a : 3, b : 4}))
    print("Div: ", sess.run(div, feed_dict={a : 3, b : 4}))
    print("Add, Sub, Mul, Div: ", sess.run([add, sub, mul, div], feed_dict={a : 3, b : 4}))