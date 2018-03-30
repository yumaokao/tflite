import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 14, 14, 4])
  x_2d = tf.fake_quant_with_min_max_args(x_2d, min=-1.0, max=3.0, num_bits=8)
  pad_value = [[0, 0], [1, 2], [2, 1], [0, 0]]
  y = tf.pad(x_2d, pad_value, "CONSTANT", name='ys')
  y = tf.fake_quant_with_min_max_args(y, min=-1.0, max=3.0, num_bits=8, name='ys')
  return y, variables

def model_tf(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8, name='ys')

def cvt_x(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

