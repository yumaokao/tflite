import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  initial = tf.truncated_normal([10, 28, 28, 1], stddev=0.1)
  W = tf.Variable(initial, name='W')
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.multiply(x_2d, W, name='ys')
  variables['W'] = W
  return y, variables
