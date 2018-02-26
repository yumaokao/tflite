import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 14, 14, 4])
  W = weight_variable([3, 3, 4, 32], name='W')
  b = bias_variable([32], name='b')

  pad_value = [[0, 0], [7, 6], [8, 7], [0, 0]]
  pad_x2d = tf.pad(x_2d, pad_value, "CONSTANT")
  x_dconv2d = tf.nn.conv2d(pad_x2d, W, strides=[1, 1, 1, 1], padding='VALID')
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
