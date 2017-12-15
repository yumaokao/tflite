import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  y = tf.nn.softmax(x, name='ys')
  return y, variables
