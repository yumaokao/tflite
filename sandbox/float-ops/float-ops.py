from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import imp

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

FLAGS = None


def main(args, op):
  dirname = os.path.dirname(os.path.abspath(__file__))
  dirname = os.path.join(dirname, op)
  exportbase = os.path.join(dirname, "export")
  if not os.path.isdir(dirname):
    raise NameError('could not find dir {}'.format(dirname))

  # import
  plugin_name = os.path.join(dirname, '{}.py'.format(op))
  if not os.path.isfile(plugin_name):
    raise NameError('could not find file {}'.format(plugin_name))
  plugin = imp.load_source('model', plugin_name)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y, variables = plugin.model(x)

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
  if args.mnist:
    mnist = input_data.read_data_sets(args.mnist_dir, one_hot=True)
    batch_xs = mnist.test.images[0:10]
  else:
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
  parser.add_argument('--mnist_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--mnist', action='store_true', help='use mnist test data')
  parser.add_argument('operations', type=str, nargs='+', help='operations to run')
  args, unparsed = parser.parse_known_args()

  for op in args.operations:
    main(args, op)
