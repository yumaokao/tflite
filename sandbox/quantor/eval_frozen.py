from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf

from prepare import prepare_dataset, prepare_metrics, prepare_tfrecords


tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'summary_dir', None, 'The directory where summaries save.')
tf.app.flags.DEFINE_string(
    'frozen_pb', None, 'The GraphDef file are stored with freeze_graph.')
tf.app.flags.DEFINE_string(
    'input_node_name', 'input', 'The name of the input node.')
tf.app.flags.DEFINE_string(
    'output_node_name', None, 'The name of the output node.')
tf.app.flags.DEFINE_integer(
    'input_size', 299, 'The width/height of the input image.')
FLAGS = tf.app.flags.FLAGS


def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.frozen_pb:
    raise ValueError('You must supply the frozen pb with --frozen_pb')
  if not FLAGS.output_node_name:
    raise ValueError('You must supply the output node name with --output_node_name')

  tf.logging.set_verbosity(tf.logging.INFO)
  tfrecords = prepare_tfrecords(FLAGS.dataset_name, FLAGS.dataset_dir,
                                FLAGS.dataset_split_name)

  if FLAGS.max_num_batches:
    num_batches = FLAGS.max_num_batches
  else:
    num_records = sum([len(list(tf.python_io.tf_record_iterator(r)))
                       for r in tfrecords])
    num_batches = int(math.ceil(num_records / float(FLAGS.batch_size)))

  #  for example in tf.python_io.tf_record_iterator(tfrecords[0]):
	#  result = tf.train.Example.FromString(example)
	#  print(result)
	#  break

  tf.logging.info('Prepare Dataset from tfrecord[0] '.format(tfrecords[0]))
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = prepare_dataset(filenames, FLAGS.dataset_name, FLAGS.input_size,
                            batch_size=FLAGS.batch_size,
                            labels_offset=FLAGS.labels_offset)
  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()

  tf.logging.info('Load GraphDef from frozen_pb {}'.format(FLAGS.frozen_pb))
  graph_def = load_graph_def(FLAGS.frozen_pb)

  tf.logging.info('Prepare metrics')
  (lbls, preds,accuracy,
   acc_update_op) = prepare_metrics(FLAGS.dataset_name,
                                    labels_offset=FLAGS.labels_offset)

  if FLAGS.summary_dir:
    tf.logging.info('Prepare summary writer')
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)
    summaries = tf.summary.merge_all()

  # Initialize `iterator` with training data.
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer, feed_dict={filenames: tfrecords})

    tf.import_graph_def(graph_def, name='')
    graph = sess.graph

    # get x and y
    x = graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))

    for step in range(num_batches):
      if (step % 100) == 0:
        print('{}/{} with batch_size {}'.format(step, num_batches,
                                                FLAGS.batch_size))
        print('  Accuracy: [{:.4f}]'.format(sess.run(accuracy)))
      images, labels = sess.run(next_batch)
      ys = sess.run(y, feed_dict={x: images})
      sess.run(acc_update_op, feed_dict={lbls: labels, preds: np.squeeze(ys)})
      if FLAGS.summary_dir:
        summary = sess.run(summaries)
        summary_writer.add_summary(summary, step)

    print('Accuracy: [{:.4f}]'.format(sess.run(accuracy)))
    # import ipdb
    # ipdb.set_trace()
    if FLAGS.summary_dir:
      summary_writer.add_graph(sess.graph)


if __name__ == '__main__':
  tf.app.run()
