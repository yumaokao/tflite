import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 14, 14, 4])
  W = weight_variable([3, 3, 8, 4], name='W')
  b = bias_variable([32], name='b')
  y = tf.nn.conv2d_transpose(x_2d, W, output_shape=[10, 28, 28, 8],
                             strides=[1, 2, 2, 1], padding='SAME',
                             name='ys')
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
