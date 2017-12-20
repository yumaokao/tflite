import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  # y = tf.concat([x, x], 0, name='ys')
  W = tf.Variable(tf.zeros([10, 28, 28, 1]), name='W')
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.concat([x_2d, W], 1, name='ys')
  variables['W'] = W
  return y, variables
