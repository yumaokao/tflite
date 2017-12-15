import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.nn.max_pool(x_2d, ksize=[1, 2, 2, 1],
                     strides=[1, 2, 2, 1], padding='SAME', name='ys')
  return y, variables
