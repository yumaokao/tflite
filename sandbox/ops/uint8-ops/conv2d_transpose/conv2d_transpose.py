import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 14, 14, 4])
  W = weight_variable([3, 3, 8, 4], name='W')
  b = bias_variable([32], name='b')
  variables['W'] = W
  variables['b'] = b


  W2 = tf.fake_quant_with_min_max_args(W, min=-1.0, max=1.0, num_bits=8)
  b2 = tf.fake_quant_with_min_max_args(W, min=-0.4, max=0.4, num_bits=8)
  y = tf.nn.conv2d_transpose(x_2d, W, output_shape=[10, 28, 28, 8],
                             strides=[1, 2, 2, 1], padding='SAME')
  y = tf.fake_quant_with_min_max_args(y, min=0.0, max=108.4, name='ys')
  return y, variables

def weight_variable(shape, name=None):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.5)
  pnames = {}
  if name is not None:
    pnames['name'] = name
  return tf.Variable(initial, **pnames)

def bias_variable(shape, name=None):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.2)
  pnames = {}
  if name is not None:
    pnames['name'] = name
  return tf.Variable(initial, **pnames)

def cvt_x(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

def model_tf(x):
  return tf.quantize_v2(x, 0.0, 108.4, tf.quint8, name='ys')
