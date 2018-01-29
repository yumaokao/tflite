import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  y = tf.nn.relu(x, name='ys')
  return y, variables
