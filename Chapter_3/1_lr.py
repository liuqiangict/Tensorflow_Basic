
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3

plt.plot(train_x, train_y, 'ro', label='Original data')
plt.legend()
plt.show()


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.zeros([1]), name='bias')

z = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_mean(tf.square(Y - z))
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
    plotdata = {'batchsize':[], 'loss':[]}
    for epoch in range(training_epoch):
        for (x, y) in zip(train_x, train_y):
            sess.run([optimizer], feed_dict={X : x, Y : y})
        if epoch % diplay_step == 0:
            loss = sess.run(cost, feed_dict={X : train_x, Y : train_y})
            print('Epoch: ', epoch + 1, ' Cost=', loss, ' W=', sess.run(W), ' b=', sess.run(b))
            if loss != 'NA':
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
    print('Finished!')
    print('Cost=',loss, ' W=', sess.run(W), ' b=', sess.run(b))
    
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fittedline') 
    plt.legend()
    plt.show()

    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs Training loss')
    plt.show()