import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  initial_W = tf.truncated_normal([784, 10], stddev=0.1)
  initial_b = tf.truncated_normal([10], stddev=0.1)
  W = tf.Variable(initial_W, name='W')
  b = tf.Variable(initial_b, name='b')
  y = tf.add(tf.matmul(x, W), b, name='ys')
  variables['W'] = W
  variables['b'] = b
  return y, variables
