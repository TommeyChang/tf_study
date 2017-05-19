import tensorflow as tf
import numpy as np

import ulib as lib

_default_weight_norm = False
_weights_stddev = None


def enable_default_weight_norm():
    global _default_weight_norm
    _default_weight_norm = True


def disable_default_weight_norm():
    global _default_weight_norm
    _default_weight_norm = False


def set_weights_stddev(weights_dev):
    global _weights_stddev
    _weights_stddev = weights_dev


def unset_weights_stddev():
    global _weights_stddev
    _weights_stddev = None


def linear(name, input_dim, output_dim, inputs,
           bias=True, init=None, weight_norm=None, gain=1.):
    """
    :param name:
    :param input_dim:
    :param output_dim:
    :param inputs:
    :param bias:
    :param init:
    :param weight_norm:
    :param gain:
    :return:
    """
    with tf.name_scope(name) as scope:

        def uniform(stddev, size):
            if _weights_stddev is not None:
                stddev = _weights_stddev
            return np.random.uniform(low=-stddev * np.sqrt(3),
                                     high=stddev * np.sqrt(3),
                                     size=size).astype(np.float32)

        if init == 'lecun':
            weight_value = uniform(np.sqrt(1. / input_dim),
                                   (input_dim, output_dim))
        elif (init == 'glorot') or (init is None):
            weight_value = uniform(np.sqrt(2. / (input_dim + output_dim)),
                                   (input_dim, output_dim))
        elif init == 'he':
            weight_value = uniform(np.sqrt(2. / input_dim),
                                   (input_dim, output_dim))
        elif init == 'glorot_he':
            weight_value = uniform(np.sqrt(4. / (input_dim + output_dim)),
                                   (input_dim, output_dim))
        elif init == 'orthogonal' or ((init is None) and (input_dim == output_dim)):
            # from lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError('Only shapes of length 2 or more are supported.')
                flat_shape = (shape[0], np.prod(shape[1:]))
                x = np.random.normal((.0, 1., flat_shape))
                u, _, v = np.linalg.svd(x, full_matrices=False)
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype(np.float32)
            weight_value = sample(shape=(input_dim, output_dim))
        elif init[0] == 'uniform':
            weight_value = np.random.uniform(low=init[1],
                                             high=init[1],
                                             size=(input_dim, output_dim)).astype(np.float32)
        else:
            raise Exception('Invalid initialization!')

        weight_value *= gain

        weight = lib.param(name + '.w', weight_value)

        if weight_norm is None:
            weight_norm = _default_weight_norm
        if weight_norm:
            norm_values = np.sqrt(np.sum(np.square(weight_value), axis=0))

            target_norms = lib.param(name + '.g', norm_values)

            with tf.name_scope('weight_norm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight *= (target_norms / norms)

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            input_reshape = tf.reshape(inputs, [-1, input_dim])
            raw_result = tf.matmul(input_reshape, weight)

            try:
                result = tf.reshape(raw_result, tf.stack(tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))
            except AttributeError:
                result = tf.reshape(raw_result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        if bias:
            result = tf.nn.bias_add(result,
                                    lib.param(name + '.b',
                                              np.zeros((output_dim, ), dtype=np.float32)))

        return result
