import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  y = tf.nn.relu6(x, name='ys')
  return y, variables
