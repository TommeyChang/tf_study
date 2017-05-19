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


def conv2d(name, input_dim, output_dim, filter_size, inputs,
           he_init=True, mask_type=None, stride=1, weight_norm=False, bias=True, gain=1.):
    """
    :param name: name space
    :param input_dim:
    :param output_dim:
    :param filter_size:
    :param inputs:
    :param he_init:
    :param mask_type: mask_type: one of None, 'a', 'b'
    :param stride:
    :param weight_norm:
    :param bias:
    :param gain:
    :return:
    """

    with tf.name_scope(name) as local_scope:

        def uniform(stddev, size):
            return np.random.uniform(low=-stddev * np.sqrt(3),
                                     high=stddev * np.sqrt(3),
                                     size=size).astype(np.float32)

        fan_in = input_dim * (filter_size ** 2)
        fan_out = output_dim * (filter_size ** 2) / (stride ** 2)

        if mask_type is not None:
            fan_in /= 2
            fan_out /= 2

        if he_init:
            filter_stddev = np.sqrt(4. / (fan_in + fan_out))
        else:
            filter_stddev = np.sqrt(2. / (fan_in + fan_out))

        if _weights_stddev is not None:
            filter_values = uniform(_weights_stddev,
                                    (filter_size, filter_size, input_dim, output_dim))
        else:
            filter_values = uniform(filter_stddev,
                                    (filter_size, filter_size, input_dim, output_dim))

        filter_values *= gain

        filters = lib.param(name + '.Filters', filter_values)

        if weight_norm is None:
            weight_norm = _default_weight_norm

        if weight_norm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0, 1, 2)))
            target_norm = lib.param(name + '.g', norm_values)
            with tf.name_scope('weight_norm') as weight_scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0, 1, 2]))
                filters *= tf.expand_dims(target_norm / norms, 1)

        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones((filter_size, filter_size, input_dim, output_dim), dtype=np.float32)
            center = filter_size // 2

            # mask out future locations
            # filter shape is (height, width, input channels, output channels
            mask[center+1, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type == 'a' and i >= j) or (mask_type == 'b' and i > j):
                        mask[center, center, i::mask_n_channels, j::mask_n_channels] = 0.

            with tf.name_scope('filter_mask'):
                filters *= mask

        result = tf.nn.conv2d(
            input=inputs,
            filter=filters,
            strides=[1, stride, stride, 1],
            padding='SAME',
            data_format='NHWC',
        )

        if bias:
            _biases = lib.param(name + '.biases', np.zeros(output_dim, dtype=np.float32))
            result = tf.nn.bias_add(result, _biases, data_format='NHWC')

        return result
