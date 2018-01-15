import numpy as np
import tensorflow as tf

def model(x): # float-in, float-out
  variables = {}
  y = tf.nn.softmax(x, name='ys')
  y = tf.fake_quant_with_min_max_args(y, min=0.0, max=1.0, num_bits=8, name='ys')
  return y, variables

def model_tf(x):
  return tf.quantize_v2(x, 0.0, 1.0, tf.quint8, name='ys')

def cvt_x(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

