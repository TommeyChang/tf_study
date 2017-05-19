import tensorflow as tf
import numpy as np

import ulib as lib

_default_weight_norm = False
_weights_stddev = None


def enable_default_weight_norm():
    global _default_weight_norm
    _default_weight_norm = True


def set_weights_stddev(weights_dev):
    global _weights_stddev
    _weights_stddev = weights_dev


def unset_weights_stddev():
    global _weights_stddev
    _weights_stddev = None


def deconv2d(name, input_dim, output_dim, filter_size, inputs,
             he_init=True, weight_norm=None, bias=True, gain=1., mask_type=None):
    """
    :param name:
    :param input_dim:
    :param output_dim:
    :param filter_size:
    :param inputs:
    :param he_init:
    :param weight_norm:
    :param bias:
    :param gain:
    :param mask_type:
    :return:
    """
    with tf.name_scope(name) as local_scope:

        if mask_type is not None:
            raise Exception('Unsupport configuration!')

        def uniform(stddev, size):
            return np.random.uniform(low=-stddev * np.sqrt(3),
                                     high=stddev * np.sqrt(3),
                                     size=size).astype(np.float32)

        stride = 2
        fan_in = input_dim * (filter_size ** 2)
        fan_out = output_dim * (filter_size ** 2) / (stride ** 2)

        if he_init:
            filter_stddev = np.sqrt(4. / (fan_in + fan_out))
        else:
            filter_stddev = np.sqrt(2. / (fan_in + fan_out))

        if _weights_stddev is not None:
            filter_values = uniform(_weights_stddev,
                                    (filter_size, filter_size, output_dim, input_dim))
        else:
            filter_values = uniform(filter_stddev,
                                    (filter_size, filter_size, output_dim, input_dim))

        filter_values *= gain

        filters = lib.param(name + '.Filters', filter_values)

        if weight_norm is None:
            weight_norm = _default_weight_norm

        if weight_norm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0, 1, 3)))
            target_norm = lib.param(name + '.g', norm_values)
            with tf.name_scope('weight_norm') as weight_scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0, 1, 3]))
                filters *= tf.expand_dims(target_norm / norms, 1)

        inputs = tf.transpose(inputs, [0, 2, 3, 1], name='NCHW2NHWC')

        input_shape = tf.shape(inputs)
        try:
            output_shape = tf.pack([input_shape[0], 2 * input_shape[1], 2 * input_shape[1], output_dim])
        except AttributeError:
            output_shape = tf.stack([input_shape[0], 2 * input_shape[1], 2 * input_shape[1], output_dim])

        result = tf.nn.conv2d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding='SAME'
        )

        if bias:
            _biases = lib.param(name + '.biases', np.zeros(output_dim, dtype=np.float32))
            result = tf.nn.bias_add(result, _biases)

        result_r = tf.transpose(result, [0, 3, 1, 2], name='NHWC2NCHW')

        return result_r

