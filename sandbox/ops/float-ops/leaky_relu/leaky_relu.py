import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  y = tf.nn.leaky_relu(x)
  y = tf.identity(y, name='ys')
  return y, variables
