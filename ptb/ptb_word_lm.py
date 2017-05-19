import inspect
import time

import numpy as np
import tensorflow as tf

import ptb.data_reader as reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    'Model', 'Small',
    'A type or model. Possible options are: small medium, large.'
)
flags.DEFINE_string(
    'Data_path', 'None',
    'Where the training/test data is stored.'
)
flags.DEFINE_string()
