from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

FLAGS = None


def load_cats_dataset(catsdir):
  def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized

  def _preprocess_function(image):
    image_preprocess = image;
    image_preprocess /= 255;
    image_preprocess -= 0.5;
    image_preprocess *= 2.0;
    return image_preprocess

  filenames = [os.path.join('images', 'cat_{}.jpg'.format(i)) for i in range(10)]
  filenames = tf.constant(filenames)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.map(_parse_function)
  dataset = dataset.map(_preprocess_function)
  return dataset


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

  dataset = load_cats_dataset(args.catsdir)
  batched_dataset = dataset.batch(10)
  iterator = batched_dataset.make_one_shot_iterator()
  next_element = iterator.get_next()

  with tf.Session() as sess:
    tf.import_graph_def(graph_def)
    graph = sess.graph

    # get x and y
    x = graph.get_tensor_by_name('import/{}:0'.format(args.x))
    y = graph.get_tensor_by_name('import/{}:0'.format(args.y))

    # Summary
    summary_writer = tf.summary.FileWriter(summarydir)
    summary_writer.add_graph(sess.graph)

    # sess.run
    batch_xs = sess.run(next_element)
    ys = sess.run(y, feed_dict={x: batch_xs})

    import ipdb
    ipdb.set_trace()
    print(ys)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--x', type=str, default='input',
                      help='input tensor name')
  parser.add_argument('--y', type=str,
                      default='MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd',
                      help='output tensor name')
  parser.add_argument('--catsdir', type=str,
                      default='images', help='directory to cat_*.jpg')
  parser.add_argument('frozen_pb', type=str, nargs=1, help='Frozen graph file (.pb) to run')
  args, unparsed = parser.parse_known_args()

  main(args)
