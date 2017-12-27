from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

FLAGS = None


def load_graph_def(pb):
  # read pb
  with tf.gfile.GFile(args.frozen_pb[0], "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  
  return graph_def


def main(args):
  dirname = os.path.dirname(os.path.abspath(__file__))
  exportbase = os.path.join(dirname, "export")
  summarydir = os.path.join(dirname, 'summary')
  if not os.path.isdir(dirname):
    raise NameError('could not find dir {}'.format(dirname))

  # load graph
  graph_def = load_graph_def(args.frozen_pb[0])


  with tf.Session() as sess:
    tf.import_graph_def(graph_def)
    graph = sess.graph
    # for op in graph.get_operations():
    #   print(op)

    # get x and y
    x = graph.get_tensor_by_name('import/{}:0'.format(args.x))
    y = graph.get_tensor_by_name('import/{}:0'.format(args.y))

    # Summary
    summary_writer = tf.summary.FileWriter(summarydir)
    summary_writer.add_graph(sess.graph)

    # sess.run
    batch_xs = (np.random.rand(10, 224, 224, 3) - 0.5) * 4
    batch_xs = batch_xs.astype('float32')
    ys = sess.run(y, feed_dict={x: batch_xs})

  np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs)
  np.save(os.path.join(exportbase, 'ys.npy'), ys)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--x', type=str, default='input',
                      help='input tensor name')
  parser.add_argument('--y', type=str,
                      default='MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd',
                      help='output tensor name')
  parser.add_argument('frozen_pb', type=str, nargs=1, help='Frozen graph file (.pb) to run')
  args, unparsed = parser.parse_known_args()

  main(args)
