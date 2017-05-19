import time

import numpy as np
import tensorflow as tf

import ulib.ops.linear as linear_op
import ulib.ops.deconv2d as decov2d_op
import ulib.ops.conv2d as conv2d_op
import ulib.ops.relu as relu_op
import ulib.mnist_data as data_fetch
import ulib.plot as uplt
import ulib.image_saver as image_saver
import ulib as lib


def generator(_batch_size, output_channel=1, m_dim=64, _noise_dim=128, filter_size=5, noise=None):

    if noise is None:
        noise = tf.random_normal([_batch_size, _noise_dim])

    h_fc1 = linear_op.linear('generator.input', _noise_dim, 4 * 4 * 4 * m_dim, noise)
    h_fc1_act = tf.nn.relu(h_fc1)
    h_fc1_reshape = tf.reshape(h_fc1_act, [-1, 4 * m_dim, 4, 4])

    h_deconv1 = decov2d_op.deconv2d('generator.deconv1', 4 * m_dim, 2 * m_dim, filter_size, h_fc1_reshape)
    h_deconv1_act = tf.nn.relu(h_deconv1)

    h_truncate = h_deconv1_act[:, :, :7, :7]

    h_deconv2 = decov2d_op.deconv2d('generator.deconv2', 2 * m_dim, m_dim, filter_size, h_truncate)
    h_deconv2_act = tf.nn.relu(h_deconv2)

    h_deconv3 = decov2d_op.deconv2d('generator.deconv3', m_dim, output_channel, filter_size, h_deconv2_act)
    h_deconv3_act = tf.nn.relu(h_deconv3)

    return tf.reshape(h_deconv3_act, shape=[-1, 784])


def critic(inputs, input_channel=1, m_dim=64, filter_size=5, stride=2):

    inputs_r = tf.reshape(inputs, [-1, 1, 28, 28])

    inputs_t = tf.transpose(inputs_r, [0, 2, 3, 1], name='NCHW_to_NHWC')

    conv1 = conv2d_op.conv2d('critic.conv1', input_channel, m_dim, filter_size, inputs_t, stride=stride)
    conv1_act = relu_op.leaky_relu(conv1)

    conv2 = conv2d_op.conv2d('critic.conv2', m_dim, 2 * m_dim, filter_size, conv1_act, stride=stride)
    conv2_act = relu_op.leaky_relu(conv2)

    conv3 = conv2d_op.conv2d('critic.conv3', 2 * m_dim, 4 * m_dim, filter_size, conv2_act, stride=stride)
    conv3_act = relu_op.leaky_relu(conv3)

    output_r = tf.reshape(conv3_act, [-1, 4 * 4 * 4 * m_dim])
    output = linear_op.linear('critic.output', 4 * 4 * 4 * m_dim, 1, output_r)

    return tf.reshape(output, [-1])


def train_op(latent_data, real_data, _batch_size, _noise_dim=128, hyper_lambda=10):

    fake_data = generator(_batch_size=_batch_size, noise=latent_data, _noise_dim=_noise_dim)

    # critic the data with critic
    real_critic = critic(real_data)
    fake_critic = critic(fake_data)

    # get the penalty
    alpha = tf.random_uniform(shape=(_batch_size, 1), minval=0., maxval=1.)
    interpolates = real_data + alpha * (fake_data - real_data)
    inter_grad = tf.gradients(critic(interpolates), [interpolates])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(inter_grad), reduction_indices=[1]))
    grad_penalty = hyper_lambda * tf.reduce_mean(tf.square(grad_l2 - 1))

    # get the cost of generator and critic
    gen_cost = -tf.reduce_mean(fake_critic)
    crit_cost = tf.reduce_mean(fake_critic) - tf.reduce_mean(real_critic) + grad_penalty

    # extract the parameters of generator and critic
    gen_params = lib.params_with_name('generator')
    crit_params = lib.params_with_name('critic')

    # construct the train operations
    _gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,
                                           beta1=0.5,
                                           beta2=0.9).minimize(gen_cost, var_list=gen_params)
    _crit_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,
                                            beta1=0.5,
                                            beta2=0.9).minimize(crit_cost, var_list=crit_params)

    return _gen_train_op, _crit_train_op, crit_cost


def result_visual(sample_num, noise_dim):
    fixed_noise = tf.constant(np.random.normal(size=(sample_num, noise_dim)).astype('float32'))
    fixed_noise_samples = generator(128, noise=fixed_noise)
    return fixed_noise_samples


def inf_train_gen(_train_gen):
    while True:
        for images, targets in _train_gen():
            yield images


def inf_noise_gen(_batch_size, _noise_dim):
    while True:
        yield np.random.normal(loc=0, scale=1.0, size=(_batch_size, _noise_dim))


if __name__ == '__main__':
    batch_size = 50
    critic_time = 5
    h_lambda = 10
    train_iter = 200000
    image_size = 784
    visual_num = 128
    n_dim = 128

    # fetch data
    train_gen, dev_gen, test_gen = data_fetch.load(batch_size, batch_size)
    # define data and noise iter
    data_iter = inf_train_gen(train_gen)
    noise_iter = inf_noise_gen(batch_size, n_dim)

    # define latent and real data placeholder
    latent_data_ph = tf.placeholder(tf.float32, shape=(None, n_dim))
    real_data_ph = tf.placeholder(tf.float32, shape=(None, image_size))

    # construct the training graph of generator and critic, and critic loss compute and samples generate
    gen_train_op, crit_train_op, critic_loss = train_op(latent_data=latent_data_ph,
                                                        real_data=real_data_ph,
                                                        _batch_size=batch_size,
                                                        _noise_dim=n_dim)
    visual_op = result_visual(visual_num, n_dim)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in xrange(train_iter):
            start_time = time.time()

            # training the critic and get the loss
            for n in xrange(critic_time):
                real_batch = data_iter.next()
                noise_batch = noise_iter.next()
                critic_cost, _ = sess.run([critic_loss, crit_train_op],
                                          feed_dict={latent_data_ph: noise_batch,
                                                     real_data_ph: real_batch})

            uplt.plot('Train critic cost', critic_cost)
            uplt.plot('Time', time.time() - start_time)

            # training the generator
            noise_batch = noise_iter.next()
            sess.run(gen_train_op, feed_dict={latent_data_ph: noise_batch})
            # computer the dev loss and generate the sample pictures
            if epoch % 100 == 99:
                noise_batch = noise_iter.next()
                dev_critic_cost = []
                for image, _ in dev_gen():
                    _dev_critic_cost = sess.run(critic_loss,
                                                feed_dict={real_data_ph: image,
                                                           latent_data_ph: noise_batch})
                    dev_critic_cost.append(_dev_critic_cost)
                uplt.plot('Dev critic cost', np.mean(dev_critic_cost))

                gen_samples = sess.run(visual_op)
                image_saver.save_images(gen_samples.reshape((visual_num, 28, 28)), 'samples_{}.png'.format(epoch))

            if (epoch < 5) or (epoch % 10 == 9):
                lib.plot.flush()

            uplt.tick()
