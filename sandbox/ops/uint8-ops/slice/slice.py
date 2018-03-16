import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x = tf.reshape(x, [2, 14, 56, 5])
  y = tf.slice(x, [1, 7, 21, 0], [1, 6, 1, 5])
  y = tf.fake_quant_with_min_max_args(y, min=-1.0, max=3.0, name='ys')
  return y, variables

def cvt_x(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

def model_tf(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8, name='ys')
