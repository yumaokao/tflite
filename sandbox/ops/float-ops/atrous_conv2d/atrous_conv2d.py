import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 5])
  W = weight_variable([3, 3, 5, 32], name='W')
  b = bias_variable([32], name='b')
  rate = 2
  x_aconv2d = tf.nn.atrous_conv2d(x_2d, W, rate, padding='VALID')
  y = tf.nn.relu(x_aconv2d + b)
  y = tf.identity(y, name='ys')
  variables['W'] = W
  variables['b'] = b
  return y, variables

def weight_variable(shape, name=None):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  pnames = {}
  if name is not None:
    pnames['name'] = name
  return tf.Variable(initial, **pnames)

def bias_variable(shape, name=None):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  pnames = {}
  if name is not None:
    pnames['name'] = name
  return tf.Variable(initial, **pnames)
