import tensorflow as tf
import ulib.ops.linear as linear


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def relu_layer(name, num_in, num_out, inputs):
    h_output = linear.linear(name + '.linear', num_in, num_out, inputs, init='he')
    return tf.nn.relu(h_output)


def leaky_relu_layer(name, num_in, num_out, inputs):
    h_output = linear.linear(name + '.linear', num_in, num_out, inputs, init='he')
    return leaky_relu(h_output)





