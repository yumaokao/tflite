from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf

import tensorflow.contrib.quantize as qg


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
    'frozen_pb', None, 'The GraphDef file are stored with quantized_graph.')
tf.app.flags.DEFINE_string(
    'input_node_name', 'input', 'The name of the input node.')
tf.app.flags.DEFINE_string(
    'output_node_name', None, 'The name of the output node.')
tf.app.flags.DEFINE_integer(
    'input_size', 299, 'The width/height of the input image.')
tf.app.flags.DEFINE_string(
    'output_dir', None, 'The directory to save quantized graph and checkpoints.')
FLAGS = tf.app.flags.FLAGS


def prepare_cifar10_dataset(filenames, width, height):
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

  def _preprocessing_cifarnet(image, label):
    tf.summary.image('image', tf.expand_dims(image, 0))
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(image, width, height)
    tf.summary.image('resized_image', tf.expand_dims(image, 0))
    image = tf.image.per_image_standardization(image)
    tf.summary.image('std_image', tf.expand_dims(image, 0))
    return image, label

  # YMK: use _preprocessing_imagenet [-1, 1) is easier
  #      for toco with --mean_value=127.5 --std_value=127.5
  def _preprocessing_imagenet(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

  # tf.Dataset
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)
  dataset = dataset.map(_preprocessing_imagenet)
  dataset = dataset.batch(FLAGS.batch_size)
  return dataset


def prepare_imagenet_dataset(filenames, width, height):
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
    image = tf.image.resize_bilinear(image, [width, height],
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


def prepare_tfrecords(dataset_name, dataset_dir, dataset_split_name):
  with tf.name_scope("tfrecords"):
    if dataset_name == 'imagenet':
      return [os.path.join(dataset_dir, 'validation-{:05d}-of-00128'.format(i))
              for i in range(0, 128)]
    elif dataset_name == 'cifar10':
      return [os.path.join(dataset_dir, '{}_{}.tfrecord'.format(
                           dataset_name, dataset_split_name))]
    else:
      tf.logging.error('Could not found preprocessing for dataset {}'.format(dataset_name))
      return None


def prepare_dataset(filenames, dataset_name, input_size):
  with tf.name_scope("datasets"):
    if dataset_name == 'imagenet':
      return prepare_imagenet_dataset(filenames, input_size, input_size)
    elif dataset_name == 'cifar10':
      return prepare_cifar10_dataset(filenames, 32, 32)
    else:
      tf.logging.error('Could not found preprocessing for dataset {}'.format(dataset_name))
      return None


def prepare_metrics(dataset_name):
  with tf.name_scope("metrics"):
    if dataset_name == 'imagenet':
      pred_shape = [None, 1001]
    elif dataset_name == 'cifar10':
      pred_shape = [None, 10]
    else:
      tf.logging.error('Could not found metrics for dataset {}'.format(dataset_name))
      return None
    lbls = tf.placeholder(tf.int32, [None, 1])
    preds = tf.placeholder(tf.int32, pred_shape)
    accuracy, acc_update_op = tf.metrics.accuracy(lbls, tf.argmax(preds, 1))
    tf.summary.scalar('accuracy', accuracy)
    return lbls, preds, accuracy, acc_update_op


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
  if not FLAGS.output_dir:
    raise ValueError('You must supply the output directory with --output_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  tfrecords = prepare_tfrecords(FLAGS.dataset_name, FLAGS.dataset_dir,
                                FLAGS.dataset_split_name)

  if FLAGS.max_num_batches:
    num_batches = FLAGS.max_num_batches
  else:
    num_records = sum([len(list(tf.python_io.tf_record_iterator(r)))
                       for r in tfrecords])
    num_batches = int(math.ceil(num_records / float(FLAGS.batch_size)))

  tf.logging.info('Load GraphDef from frozen_pb {}'.format(FLAGS.frozen_pb))
  graph_def = load_graph_def(FLAGS.frozen_pb)

  tf.logging.info('Quantize Graph')
  with tf.Session() as sess:
    tf.import_graph_def(graph_def, name='')
    quantized_graph = qg.create_training_graph(sess.graph)
    quantized_inf_graph = qg.create_eval_graph(sess.graph)

  # Initialize `iterator` with training data.
  with tf.Session(graph=quantized_graph) as sess:
    tf.logging.info('Prepare dataset')
    with tf.name_scope("dataset"):
      filenames = tf.placeholder(tf.string, shape=[None])
      dataset = prepare_dataset(filenames, FLAGS.dataset_name, FLAGS.input_size)
      iterator = dataset.make_initializable_iterator()
      next_batch = iterator.get_next()

    tf.logging.info('Prepare metrics')
    lbls, preds, accuracy, acc_update_op = prepare_metrics(FLAGS.dataset_name)

    tf.logging.info('Prepare Saver')
    saver = tf.train.Saver()

    if FLAGS.summary_dir:
      tf.logging.info('Prepare summary writer')
      summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

    # initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer, feed_dict={filenames: tfrecords})

    graph = sess.graph

    # get x and y
    x = graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))

    # summary all min/max variables
    # print(graph.get_collection('variables')[3].eval())
    for var in graph.get_collection('variables'):
      tf.summary.scalar(var.name, var)
    summaries = tf.summary.merge_all()

    for step in range(num_batches):
      images, labels = sess.run(next_batch)
      ys = sess.run(y, feed_dict={x: images})
      sess.run(acc_update_op, feed_dict={lbls: labels, preds: ys})
      summary = sess.run(summaries)
      if FLAGS.summary_dir:
        summary_writer.add_summary(summary, step)

    print('Accuracy: [{:.4f}]'.format(sess.run(accuracy)))
    if FLAGS.summary_dir:
      summary_writer.add_graph(graph)

    # save graph and ckpts
    saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"))
    # tf.train.write_graph(graph, FLAGS.output_dir, 'quantor.pb', as_text=False)
    tf.train.write_graph(quantized_inf_graph, FLAGS.output_dir, 'quantor.pb', as_text=False)


if __name__ == '__main__':
  tf.app.run()
