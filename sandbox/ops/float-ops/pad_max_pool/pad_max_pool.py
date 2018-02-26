import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  pad_value = [[0, 0], [1, 2], [2, 1], [0, 0]]
  pad_x2d = tf.pad(x_2d, pad_value, 'CONSTANT')
  y = tf.nn.max_pool(pad_x2d, ksize=[1, 2, 2, 1],
                     strides=[1, 2, 2, 1], padding='VALID', name='ys')
  return y, variables
