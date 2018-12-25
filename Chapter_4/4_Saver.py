
import tensorflow as tf

# !.0 Save model
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "save_path/file_name")


# 2.0 Restore model
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "save_path/file_name")


# 3.0 Print model data
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file("save_path/file_name", None, True)