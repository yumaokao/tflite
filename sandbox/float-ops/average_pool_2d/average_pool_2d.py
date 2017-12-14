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

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

FLAGS = None


def main(_):
  # dir base
  dirname = os.path.dirname(os.path.abspath(__file__))
  exportbase = os.path.join(dirname, "export")

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  x_2d = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.nn.avg_pool(x_2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # variables initializer
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Add ops to save and restore all the variables.
  # saver = tf.train.Saver()

  # summary
  summary_writer = tf.summary.FileWriter(os.path.join(dirname, "summary"), graph=tf.get_default_graph())

  # Test initialed model
  pb_path = tf.train.write_graph(sess.graph_def, dirname, "model.pb", False)
  print("  GraphDef saved in file: %s" % pb_path)
  # ckpt_path = saver.save(sess, os.path.join(dirname, "ckpts", "model.ckpt"))
  # print("  Model saved in file: %s" % ckpt_path)

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Save a batch 10
  batch_xs = mnist.test.images[0:10]
  batch_ys = mnist.test.labels[0:10]
  # print("  batch_ys:")
  # print(batch_ys)

  # run test
  # print("  y:")
  ys = sess.run(y, feed_dict={x: batch_xs})
  # print(ys.shape)

  # save to npy
  np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs)
  np.save(os.path.join(exportbase, 'batch_ys.npy'), batch_ys)
  np.save(os.path.join(exportbase, 'ys.npy'), ys)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
