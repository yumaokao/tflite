import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  W = tf.Variable(tf.zeros([784, 10]), name='W')
  b = tf.Variable(tf.zeros([10]), name='b')
  y = tf.matmul(x, W) + b
  variables['W'] = W
  variables['b'] = b
  return y, variables
