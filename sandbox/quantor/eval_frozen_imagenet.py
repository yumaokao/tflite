from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf


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

tf.app.flags.DEFINE_string(
    'summary_dir', None, 'The directory where summaries save.')

tf.app.flags.DEFINE_string(
    'frozen_pb', None, 'The GraphDef file are stored with freeze_graph.')

tf.app.flags.DEFINE_string(
    'output_node_names', None, 'The name of the output node.')

tf.app.flags.DEFINE_integer(
    'input_size', 299, 'The width/height of the input image.')

FLAGS = tf.app.flags.FLAGS

def prepare_imagenet_dataset(filenames):
  def _read_tfrecord(example_proto):
    feature_to_type = {
        "image/class/label": tf.FixedLenFeature([1], dtype=tf.int64),
        "image/encoded": tf.FixedLenFeature([], dtype=tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    label = parsed_features["image/class/label"]
    rawpng = parsed_features["image/encoded"]
    image_decoded = tf.image.decode_png(rawpng, channels=3)
    return image_decoded, label

  def _preprocessing(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.input_size, FLAGS.input_size],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

  # tf.Dataset
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)
  dataset = dataset.map(_preprocessing)
  dataset = dataset.batch(FLAGS.batch_size)
  return dataset


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

  tf.logging.set_verbosity(tf.logging.INFO)
  tfrecord_pattern = [os.path.join(FLAGS.dataset_dir, 'validation-{:05d}-of-00128'.format(i)) for i in range(0, 128)]
  tf.logging.info('Import Dataset from tfrecord {}'.format(tfrecord_pattern))

  if FLAGS.max_num_batches:
    num_batches = FLAGS.max_num_batches
  else:
    num_records = sum([len(list(tf.python_io.tf_record_iterator(record))) for record in tfrecord_pattern])
    #num_records = len(list(tf.python_io.tf_record_iterator(tfrecord_pattern)))
    num_batches = int(math.ceil(num_records / float(FLAGS.batch_size)))

  # for example in tf.python_io.tf_record_iterator(tfrecord_pattern[0]):
	# result = tf.train.Example.FromString(example)
	# print(result)
	# break

  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = prepare_imagenet_dataset(filenames)
  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()

  tf.logging.info('Load GraphDef from frozen_pb {}'.format(FLAGS.frozen_pb))
  graph_def = load_graph_def(FLAGS.frozen_pb)

  tf.logging.info('Prepare metrics')
  with tf.name_scope("metrics"):
    lbls = tf.placeholder(tf.int32, [None, 1])
    preds = tf.placeholder(tf.int32, [None, 1001])
    accuracy, acc_update_op = tf.metrics.accuracy(lbls, tf.argmax(preds, 1))
    tf.summary.scalar('accuracy', accuracy)

  if FLAGS.summary_dir:
    tf.logging.info('Prepare summary writer')
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)
    summaries = tf.summary.merge_all()

  # Initialize `iterator` with training data.
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer, feed_dict={filenames: tfrecord_pattern})

    tf.import_graph_def(graph_def, name='')
    graph = sess.graph

    # get x and y
    x = graph.get_tensor_by_name('{}:0'.format('input'))
    y = graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_names))

    for step in range(num_batches):
      images, labels = sess.run(next_batch)
      ys = sess.run(y, feed_dict={x: images})
      sess.run(acc_update_op, feed_dict={lbls: labels, preds: ys})
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