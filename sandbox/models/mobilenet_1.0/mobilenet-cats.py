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
  def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized, label

  def _preprocess_function(image, label):
    image_preprocess = image;
    image_preprocess /= 255;
    image_preprocess -= 0.5;
    image_preprocess *= 2.0;
    return image_preprocess, label

  filenames = [os.path.join('images', 'cat_{}.jpg'.format(i)) for i in range(10)]
  filenames = tf.constant(filenames)
  labels = tf.constant([282 for i in range(10)])
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  dataset = dataset.map(_parse_function)
  dataset = dataset.map(_preprocess_function)
  return dataset


def load_graph_def(pb):
  # read pb
  with tf.gfile.GFile(args.frozen_pb[0], "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def calulate_accuray(y, y_):
  with tf.name_scope('accuracy'):
    top_1 = tf.argmax(tf.squeeze(y), 1)
    top_1 = tf.cast(top_1, tf.int32)
    correct_prediction = tf.equal(top_1, y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  return accuracy

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
    batch_xs, batch_ys = sess.run(next_element)
    ys = sess.run(y, feed_dict={x: batch_xs})

    y_ = tf.placeholder(tf.int32, [None, 1])
    accuracy = calulate_accuray(y, y_)
    cat_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys.reshape(10, 1)})
    print(cat_accuracy)

  np.save(os.path.join(exportbase, 'batch_xs.npy'), batch_xs)
  np.save(os.path.join(exportbase, 'ys.npy'), ys)


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
