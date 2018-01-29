from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

FLAGS = None

def model(x): # float-in, float-out
  variables = {}
  x_2d = tf.reshape(x, [-1, 14, 14, 4])
  x_2d = tf.fake_quant_with_min_max_args(x_2d, min=-1.0, max=3.0, num_bits=8)

  y = tf.depth_to_space(x_2d, 2)
  y = tf.fake_quant_with_min_max_args(y, min=-1.0, max=3.0, num_bits=8, name='ys')
  return y, variables

def model_tf(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8, name='ys')

def cvt_x(x):
  return tf.quantize_v2(x, -1.0, 3.0, tf.quint8)

def main(args):
  dirname = os.path.dirname(os.path.abspath(__file__))
  exportbase = os.path.join(dirname, "export")
  if not os.path.isdir(dirname):
    raise NameError('could not find dir {}'.format(dirname))

  data_num = 784
  # Create the model
  x = tf.placeholder(tf.float32, [None, data_num])
  y, variables = model(x)

  # Variables initializer
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Summary
  summary_writer = tf.summary.FileWriter(os.path.join(dirname, "summary"),
                                         graph=tf.get_default_graph())


  # Save GraphDef
  pb_path = tf.train.write_graph(sess.graph_def, dirname, "model.pb", False)
  print("  GraphDef saved in file: %s" % pb_path)

  # Save Checkpoints
  if len(variables) > 0:
    saver = tf.train.Saver()
    ckpt_path = saver.save(sess, os.path.join(dirname, "ckpts", "model.ckpt"))
    print("  Model saved in file: %s" % ckpt_path)

  # Import data
  # import ipdb
  batch_xs = (np.random.rand(10, data_num) - 0.25) * 4 # [-1.0 ~ 1.5]
  batch_xs = batch_xs.astype('float32')

  # Run test
  ys = sess.run(model_tf(y), feed_dict={x: batch_xs})
  batch_xs2 = sess.run(cvt_x(x), feed_dict={x: batch_xs})
  # ys = sess.run(y, feed_dict={x: batch_xs})
  # print('input float:\n', batch_xs)
  # print('input quant:\n', batch_xs2)
  # print('output tf_quant:\n', ys)
  # ipdb.set_trace()

  # Save to npy
  np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs2.output)
  np.save(os.path.join(exportbase, 'ys.npy'), ys.output)

  # Save weights to npy
  for k in variables:
    v = variables[k]
    np.save(os.path.join(exportbase, '{}.npy'.format(k)), v.eval())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args, unparsed = parser.parse_known_args()

  main(args)
