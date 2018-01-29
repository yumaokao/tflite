import numpy as np
import tensorflow as tf

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, mean=1.0, stddev=0.0)
  pnames = {}
  if name is not None:
    pnames['name'] = name
  return tf.Variable(initial, **pnames)

def bias_variable(shape, name=None):
  initial = tf.truncated_normal(shape, mean=0.0, stddev=0.2)
  pnames = {}
  if name is not None:
    pnames['name'] = name
  return tf.Variable(initial, **pnames)

def model(x): # float-in, float-out
  variables = {}
  # fake_x = tf.fake_quant_with_min_max_args(x, min=-32.0, max=31.0, num_bits=8)
  W = weight_variable([784, 10], name='W')
  b = bias_variable([10], name='b')
  variables['W'] = W
  variables['b'] = b
  W2 = tf.fake_quant_with_min_max_args(W, min=0.0, max=256.0, num_bits=8)
  b2 = tf.fake_quant_with_min_max_args(b, min=-0.4, max=0.4, num_bits=8)
 
  # 3 * 1.0 * 784 + 0.4 = 2352.4
  y = tf.matmul(x, W2)
  y = tf.nn.relu(tf.add(y, b2))
  y = tf.fake_quant_with_min_max_args(y, min=-0.0, max=2352.4, num_bits=8, name='ys')

  return y, variables

def model_tf(x):
  return tf.quantize_v2(x, 0.0, 2352.4, tf.quint8, name='ys')

def cvt_x(x):
  return tf.quantize_v2(x, -1, 3, tf.quint8)

