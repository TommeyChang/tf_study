import ulib as lib
import numpy as np
import tensorflow as tf


def batch_norm(name, axes, inputs,
               is_training=None, stats_iter=None, update_moving_stats=True, fused=True):

    if ((axes == [0, 2, 2]) or (axes == [0, 2])) and fused:
        if axes == [0, 2]:
            inputs = tf.expand_dims(inputs, 3)

        offset = lib.param(name + '.offset', np.zeros(inputs.get_shape()[1], dtype=np.float32))
        scale = lib.param(name + '.scale', np.ones(inputs.get_shape()[1], dtype=np.float32))

        moving_mean = lib.param(name + '.moving_mean',
                                np.zeros(inputs.get_shape()[1], dtype=np.float32),
                                trainable=False)
        moving_var = lib.param(name + '.moving_var',
                               np.ones(inputs.get_shape()[1], dtype=np.float32),
                               trainable=False)

        def _fused_batch_norm_train():
            return tf.nn.fused_batch_norm(inputs, scale=scale, offset=offset, epsilon=1e-5, data_format='NCHW')

        def _fused_batch_norm_inference():
            batch_size = tf.cast(tf.shape(inputs)[0], np.float32)
            _mean, _var = tf.nn.moments(inputs, [2, 3], keep_dims=True)
            _mean = ((1.0 / batch_size) * _mean) + ((batch_size - 1.) / batch_size * moving_mean)[None, :, None, None]
            _var = ((1.0 / batch_size) * _var) + (((batch_size - 1.) / batch_size) * moving_var)[None, :, None, None]
            return tf.nn.batch_normalization(inputs, _mean, _var,
                                             offset[None, :, None, None],
                                             scale[None, :, None, None],
                                             1e-5), _mean, _var

        if is_training is None:
            outputs, batch_mean, batch_var = _fused_batch_norm_train()
        else:
            outputs, batch_mean, batch_var = tf.cond(is_training,
                                                     _fused_batch_norm_train(),
                                                     _fused_batch_norm_inference())

            if update_moving_stats:

                no_update = lambda: outputs

                def _force_update():
                    float_stat_iter = tf.cast(stats_iter, tf.float32)

                    update_moving_mean = tf.assign(moving_mean,
                                                   ((float_stat_iter / (float_stat_iter + 1)) * moving_mean) +
                                                   ((1 / (float_stat_iter + 1)) * batch_mean))
                    update_moving_var = tf.assign(moving_var,
                                                  ((float_stat_iter / (float_stat_iter + 1)) * moving_var) +
                                                  ((1 / (float_stat_iter + 1)) * batch_var))

                    with tf.control_dependencies([update_moving_mean, update_moving_var]):
                        return tf.identity(outputs)

                outputs = tf.cond(is_training, _force_update(), no_update)
        if axes == [0, 2]:
            return outputs[:, :, :, 0]  # collapse last dim
        else:
            return outputs

    else:
        # raise Exception('old BN')
        # TODO we can probably use nn.fused_batch_norm here too for speedup
        mean, var = tf.nn.moments(inputs, axes=axes, keep_dims=True)
        shape = mean.get_shape().as_list()
        if 0 not in axes:
            print "WARNING ({}): didn't find 0 in axes, " \
                  "but not using separate BN params for each item in batch".format(name)
            shape[0] = 1
        offset = lib.param(name + '.offset', np.zeros(shape, dtype=np.float32))
        scale = lib.param(name + '.scale', np.ones(shape, dtype=np.float32))
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)
