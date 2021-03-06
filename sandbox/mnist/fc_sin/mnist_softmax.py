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

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]), name='W')
  b = tf.Variable(tf.zeros([10]), name='b')
  b0 = tf.Variable(tf.zeros([10]), name='b0')
  y = tf.sin(tf.matmul(x, W) + b) + b0

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  # summary
  summary_writer = tf.summary.FileWriter(os.path.join(dirname, "summary"), graph=tf.get_default_graph())

  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("YMK: accuracy")
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  print("YMK: save to pb and ckpt")
  pb_path = tf.train.write_graph(sess.graph_def, dirname, "mnist.pb", False)
  print("  GraphDef saved in file: %s" % pb_path)
  ckpt_path = saver.save(sess, os.path.join(dirname, "ckpts", "model.ckpt"))
  print("  Model saved in file: %s" % ckpt_path)

  # print W
  print("YMK: print details")
  print("  W:")
  print(W.eval())
  print("  b:")
  print(b.eval())
  print("  b0:")
  print(b0.eval())

  # Save a batch 10
  print("YMK: print mnist test first 10")
  batch_xs = mnist.test.images[0:10]
  batch_ys = mnist.test.labels[0:10]
  print("  batch_ys:")
  print(batch_ys)

  # run test
  print("  y:")
  ys = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
  print(ys)
  # import ipdb
  # ipdb.set_trace()

  # save to txt
  print(exportbase)
  np.save(os.path.join(exportbase, 'W.npy'), W.eval())
  np.save(os.path.join(exportbase, 'b.npy'), b.eval())
  np.save(os.path.join(exportbase, 'b0.npy'), b0.eval())
  np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs)
  np.save(os.path.join(exportbase, 'batch_ys.npy'), batch_ys)
  np.save(os.path.join(exportbase, 'ys.npy'), ys)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
