import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  y = tf.nn.tanh(x, name='ys')
  return y, variables
