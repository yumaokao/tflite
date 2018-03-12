import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x = tf.reshape(x, [2, 14, 56, 5])
  y = tf.slice(x, [1, 7, 21, 0], [1, 6, 1, 5], name='ys')
  return y, variables
