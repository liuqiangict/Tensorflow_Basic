
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.zeros([1]), name='bias')

z = tf.add(tf.multiply(X, W), b)
tf.summary.histogram('z', z)

cost = tf.reduce_mean(tf.square(Y - z))
tf.summary.scalar('loss_function', cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()

training_epoch = 20
diplay_step = 2


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

with tf.Session() as sess:
    sess.run(init)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/summary', sess.graph)

    for epoch in range(training_epoch):
        for (x, y) in zip(train_x, train_y):
            sess.run([optimizer], feed_dict={X : x, Y : y})
            summary_str = sess.run(merged_summary, feed_dict={X : x, Y : y})
            summary_writer.add_summary(summary_str, epoch)
    print('Finished!')

# > tensorboard --logdir .\log\summary