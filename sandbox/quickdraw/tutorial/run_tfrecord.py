from __future__ import print_function

import argparse
import os
import tensorflow as tf


def main(args):
  #  for example in tf.python_io.tf_record_iterator("datasets/tutorial_v1/training.tfrecord-00001-of-00010"):
    #  result = tf.train.Example.FromString(example)
    #  print(result)
    #  break

  def _read_tfrecord(example_proto):
    feature_to_type = {
        "ink": tf.VarLenFeature(dtype=tf.float32),
        "shape": tf.FixedLenFeature([2], dtype=tf.int64),
        "class_index": tf.FixedLenFeature([1], dtype=tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    # parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])
    label = parsed_features["class_index"]
    shape = parsed_features["shape"]
    ink = parsed_features["ink"]
    return ink, shape, label

  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)  # Parse the record into tensors.
  dataset = dataset.repeat()  # Repeat the input indefinitely.
  dataset = dataset.batch(32)
  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()

  # Initialize `iterator` with training data.
  with tf.Session() as sess:
    training_filenames = ["datasets/tutorial_v1/training.tfrecord-00001-of-00010"]
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    inks, shapes, labels = sess.run(next_batch)
    print(labels)
    inks, shapes, labels = sess.run(next_batch)
    print(labels)
    inks, shapes, labels = sess.run(next_batch)
    print(labels)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  args, unparsed = parser.parse_known_args()
  main(args)
