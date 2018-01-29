import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.reduce_mean(x, [-1, 1, 1, -1], name='ys')
  return y, variables
