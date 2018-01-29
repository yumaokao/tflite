import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  # tf.sqrt(tf.nn.ave_pool(tf.square(h))
  x_square = tf.square(x_2d)
  x_avgpool = tf.nn.avg_pool(x_square, ksize=[1, 2, 2, 1],
                     strides=[1, 2, 2, 1], padding='SAME')
  y = tf.sqrt(x_avgpool, name='ys')
  return y, variables
