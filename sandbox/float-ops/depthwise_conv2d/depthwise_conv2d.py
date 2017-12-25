import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  W = weight_variable([5, 5, 1, 32], name='W')
  b = bias_variable([32], name='b')
  x_dconv2d = tf.nn.depthwise_conv2d(x_2d, W, strides=[1, 1, 1, 1], padding='SAME')
  y = tf.nn.relu(x_dconv2d + b, name='ys')
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
