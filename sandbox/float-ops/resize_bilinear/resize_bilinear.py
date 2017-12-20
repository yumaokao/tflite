import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.image.resize_bilinear(x_2d, [54, 54], name='ys')
  return y, variables
