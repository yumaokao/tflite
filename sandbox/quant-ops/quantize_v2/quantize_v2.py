from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

FLAGS = None


def model(x):
  variables = {}
  y = tf.quantize_v2(x, -2.0, 2.0, tf.quint8, name='ys')
  return y, variables

def main(args):
  dirname = os.path.dirname(os.path.abspath(__file__))
  exportbase = os.path.join(dirname, "export")
  if not os.path.isdir(dirname):
    raise NameError('could not find dir {}'.format(dirname))

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
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
  batch_xs = (np.random.rand(10, 784) - 0.5) * 4
  batch_xs = batch_xs.astype('float32')

  # Run test
  ys = sess.run(y, feed_dict={x: batch_xs})

  # Save to npy
  np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs)
  np.save(os.path.join(exportbase, 'ys.npy'), ys)

  # Save weights to npy
  for k in variables:
    v = variables[k]
    np.save(os.path.join(exportbase, '{}.npy'.format(k)), v.eval())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args, unparsed = parser.parse_known_args()

  main(args)
