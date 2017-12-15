# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
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
  y = plugin.model(x)

  # Variables initializer
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Summary
  summary_writer = tf.summary.FileWriter(os.path.join(dirname, "summary"),
                                         graph=tf.get_default_graph())

  # Saver
  # saver = tf.train.Saver()

  # Save GraphDef
  pb_path = tf.train.write_graph(sess.graph_def, dirname, "model.pb", False)
  print("  GraphDef saved in file: %s" % pb_path)

  # Save Checkpoints
  # ckpt_path = saver.save(sess, os.path.join(dirname, "ckpts", "model.ckpt"))
  # print("  Model saved in file: %s" % ckpt_path)

  # Import data
  mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

  # Save a batch 10
  batch_xs = mnist.test.images[0:10]
  batch_ys = mnist.test.labels[0:10]

  # Run test
  ys = sess.run(y, feed_dict={x: batch_xs})

  # Save to npy
  np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs)
  np.save(os.path.join(exportbase, 'ys.npy'), ys)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('operations', type=str, nargs='+', help='operations to run')
  args, unparsed = parser.parse_known_args()

  for op in args.operations:
    main(args, op)
