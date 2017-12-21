import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 14, 14, 4])
  y = tf.depth_to_space(x_2d, 2, name='ys')
  return y, variables
