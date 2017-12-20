import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  y = tf.concat([x, x], 0, name='ys')
  # x_2d = tf.reshape(x, [-1, 28, 28, 1])
  # y = tf.concat([x_2d, x_2d], 1, name='ys')
  return y, variables
