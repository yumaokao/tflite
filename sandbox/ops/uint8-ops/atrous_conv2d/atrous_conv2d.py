import numpy as np
import tensorflow as tf

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, mean=0.0, stddev=0.5)
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
  x_2d = tf.reshape(x, [-1, 28, 28, 2])
  fake_x = tf.fake_quant_with_min_max_args(x_2d, min=-1.0, max=3.0, num_bits=8)
  
  W = weight_variable([3, 3, 2, 32], name='W')
  b = bias_variable([32], name='b')
  variables['W'] = W
  variables['b'] = b
  W2 = tf.fake_quant_with_min_max_args(W, min=-1.0, max=1.0, num_bits=8)
  b2 = tf.fake_quant_with_min_max_args(b, min=-0.4, max=0.4, num_bits=8)
 
  rate = 2
  x_aconv2d = tf.nn.atrous_conv2d(fake_x, W2, rate, padding='VALID')
  y = tf.nn.relu(x_aconv2d + b2)
  # 3 * 1.0 * (5 * 5 * 1) = 75
  y = tf.fake_quant_with_min_max_args(y, min=0.0, max=75.4, name='ys')
  return y, variables

def cvt_x(x):
    return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

def model_tf(x):
  # return x
  return tf.quantize_v2(x, 0.0, 75.4, tf.quint8, name='ys')

