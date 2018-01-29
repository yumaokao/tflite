import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.nn.l2_normalize(x_2d, 3, name='ys')
  return y, variables
