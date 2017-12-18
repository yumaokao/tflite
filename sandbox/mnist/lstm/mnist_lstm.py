from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

FLAGS = None


def rnn(x):
  # Return variables to save
  variables = {}

  # Reshape to 2d
  x_2d = tf.reshape(x, [-1, 28, 28])

  # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
  x_t = tf.unstack(x_2d, 28, 1)

  # Define a lstm cell with tensorflow
  lstm_cell = tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)
  import ipdb
  ipdb.set_trace()
  
  # Get lstm cell output
  outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_t, dtype=tf.float32)
  
  # Linear activation, using rnn inner loop last output
  W = tf.Variable(tf.random_normal([128, 10]), name='W')
  b = tf.Variable(tf.random_normal([10]), name='b')
  variables['W'] = W
  variables['b'] = b
  y = tf.matmul(outputs[-1], W) + b

  return y, variables


def main(_):
  # dir base
  dirname = os.path.dirname(os.path.abspath(__file__))

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, variables = rnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  # summary
  summary_writer = tf.summary.FileWriter(os.path.join(dirname, "summary"), graph=tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))

    pb_path = tf.train.write_graph(sess.graph_def, dirname, "mnist.pb", False)
    print("GraphDef saved in file: %s" % pb_path)
    ckpt_path = saver.save(sess, os.path.join(dirname, "ckpts", "model.ckpt"))
    print("Model saved in file: %s" % ckpt_path)

    # save a batch of 10
    batch_xs = mnist.test.images[0:10]
    batch_ys = mnist.test.labels[0:10]
    ys = sess.run(y_conv, feed_dict={x: batch_xs, y_: batch_ys})
    exportbase = os.path.join(dirname, "export")
    np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs)
    np.save(os.path.join(exportbase, 'batch_ys.npy'), batch_ys)
    np.save(os.path.join(exportbase, 'ys.npy'), ys)

    # save all variables into npy
    for k in variables:
        v = variables[k]
        np.save(os.path.join(exportbase, '{}.npy'.format(k)), v.eval())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
