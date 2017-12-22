import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 7, 7, 16])
  y = tf.nn.local_response_normalization(x_2d, 5, 1, 1, 0.5, name='ys')
  return y, variables
