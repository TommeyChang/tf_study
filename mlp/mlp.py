import tensorflow as tf

import data_fetch


def fetch_data():
    return data_fetch.read_data_sets('./data_set', one_hot=True)


def mnist_simple():
    images_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
    label_placeholder = tf.placeholder(tf.float32, shape=[None, 10])

    weights = tf.Variable(tf.zeros([784, 10]))
    bias = tf.Variable(tf.zeros([10]))

    prediction = tf.nn.softmax(tf.matmul(images_placeholder, weights) + bias)

    cross_entropy = -tf.reduce_sum(label_placeholder * tf.log(prediction))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    mnist = fetch_data()

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            batch_data, batch_label = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={images_placeholder: batch_data,
                                            label_placeholder: batch_label})
            if i % 100 == 99:
                print 'Epoch:', i

        print sess.run(accuracy, feed_dict={images_placeholder: mnist.test.images,
                                            label_placeholder: mnist.test.labels})


def mnist_cnn():

    def weight_variable(shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    data_ph = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    label_ph = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    weight_conv1 = weight_variable([5, 5, 1, 32])
    bias_conv1 = bias_variable([32])

    data_r = tf.reshape(data_ph, shape=[-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(data_r, weight_conv1) + bias_conv1)
    h_pool1 = max_pool(h_conv1)

    weights_conv2 = weight_variable([5, 5, 32, 64])
    bias_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights_conv2) + bias_conv2)
    h_pool2 = max_pool(h_conv2)

    weight_fc1 = weight_variable([7 * 7 * 64, 1024])
    bias_fc1 = bias_variable([1024])

    h_flatten = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_flatten, weight_fc1) + bias_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    weight_out = weight_variable([1024, 10])
    bias_out = bias_variable([10])

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, weight_out) + bias_out)

    cross_entropy = -tf.reduce_sum(label_ph * tf.log(prediction))

    train_step = tf.train.AdamOptimizer(learning_rate=0.001,
                                        beta1=0.5,
                                        beta2=0.9).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    data_set = fetch_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(20000):
            batch = data_set.train.next_batch(50)
            train_step.run(feed_dict={
                data_ph: batch[0],
                label_ph: batch[1],
                keep_prob: 0.5
            })

            if i % 100 == 99:
                train_acc = accuracy.eval(feed_dict={
                    data_ph: batch[0],
                    label_ph: batch[1],
                    keep_prob: 1.0
                })
                print "Epoch: %d, training accuracy: %g" % (i + 1, train_acc)

        print 'Test accuracy: %g' % accuracy.eval(feed_dict={
            data_ph: data_set.test.images,
            label_ph: data_set.test.labels,
            keep_prob: 1.0
        })


if __name__ == '__main__':
    mnist_cnn()




