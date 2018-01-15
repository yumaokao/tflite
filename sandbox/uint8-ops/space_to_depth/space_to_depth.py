import numpy as np
import tensorflow as tf

def model(x): # float-in, float-out
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  x_2d = tf.fake_quant_with_min_max_args(x_2d, min=-1.0, max=3.0, num_bits=8)

  y = tf.space_to_depth(x_2d, 2)
  y = tf.fake_quant_with_min_max_args(y, min=-1.0, max=3.0, num_bits=8, name='ys')
  return y, variables

def model_tf(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8, name='ys')

def cvt_x(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)
