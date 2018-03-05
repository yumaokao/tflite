import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 14, 14, 4])
  pad_value = [[0, 0], [1, 1], [1, 1], [0, 0]]
  y = tf.pad(x_2d, pad_value, "CONSTANT", name='ys')
  # x_dconv2d = tf.nn.conv2d(x_2d, W, strides=[1, 1, 1, 1], padding='SAME')
  # y = tf.nn.relu(x_dconv2d + b, name='ys')
  # variables['W'] = W
  # variables['b'] = b
  return y, variables

