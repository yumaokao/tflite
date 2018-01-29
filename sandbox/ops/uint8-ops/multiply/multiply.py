import numpy as np
import tensorflow as tf

def generate_variable(shape, name=None):
  initial = tf.truncated_normal(shape, mean=1.0, stddev=1.0)
  pnames = {}
  if name is not None:
    pnames['name'] = name
  return tf.Variable(initial, **pnames)


def model(x): # float-in, float-out
  variables = {}
  
  W = generate_variable([10, 28, 28, 1], name='W')
  variables['W'] = W
  W2 = tf.fake_quant_with_min_max_args(W, min=-1.0, max=3.0, num_bits=8)
 
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  x_2d = tf.fake_quant_with_min_max_args(x_2d, min=-1.0, max=3.0, num_bits=8)

  y = tf.multiply(x_2d, W2)
  y = tf.fake_quant_with_min_max_args(y, min=-3.0, max=9.0 ,name='ys')
  return y, variables

def cvt_x(x):
    return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

def model_tf(x):
  # return x
  return tf.quantize_v2(x, -3.0, 9.0, tf.quint8, name='ys')

