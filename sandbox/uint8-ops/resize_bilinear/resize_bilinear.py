import tensorflow as tf

def model(x):
  # return variables to save
  variables = {}
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.image.resize_bilinear(x_2d, [54, 54])
  y = tf.fake_quant_with_min_max_args(y, min=0.0, max=3.0, name='ys')
  return y, variables

def cvt_x(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

def model_tf(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8, name='ys')
