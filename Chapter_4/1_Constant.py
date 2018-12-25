
import tensorflow as tf

stringCon = tf.constant("Hello Tensorflow.", dtype=tf.string, name='HelloTensorflow') # => b'Hello Tensorflow.'

arrayCon1 = tf.constant([1, 2, 3, 4, 5, 6, 7]) # => [1 2 3 4 5 6 7]

arrayCon2 = tf.constant(-1.0, shape=[2, 3]) # => [[-1. -1. -1.], [-1. -1. -1.]]

sess = tf.Session()
print(sess.run(stringCon))
print(sess.run(arrayCon1))
print(sess.run(arrayCon2))