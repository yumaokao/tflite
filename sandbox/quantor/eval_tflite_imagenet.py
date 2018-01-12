from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tempfile
import subprocess
import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

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
    'tflite_model', None, 'The TFLite model file is stored with toco.')

tf.app.flags.DEFINE_string(
    'inference_type', 'float', 'The inference type to run the tflie model')

tf.app.flags.DEFINE_string(
    'tensorflow_dir', 'string', 'The directory where the tensorflow are stored')

tf.app.flags.DEFINE_integer(
    'input_size', 299, 'The width/height of the input image.')


FLAGS = tf.app.flags.FLAGS


def prepare_imagenet_dataset(filenames, inference_type):
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
    image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
    tf.summary.image('resized_image', tf.expand_dims(image, 0))
    image = tf.image.per_image_standardization(image)
    tf.summary.image('std_image', tf.expand_dims(image, 0))
    return image, label

  # YMK: use _preprocessing_imagenet [-1, 1) is easier
  #      for toco with --mean_value=127.5 --std_value=127.5
  def _preprocessing_imagenet(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.input_size, FLAGS.input_size],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

  def _resize_imagenet(image, label):
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.input_size, FLAGS.input_size],
                                     align_corners=False)
    return image, label


  # tf.Dataset
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)
  if inference_type == 'float':
    dataset = dataset.map(_preprocessing_imagenet)
  else:
    dataset = dataset.map(_resize_imagenet)
  dataset = dataset.batch(FLAGS.batch_size)
  return dataset


def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def prepare_run_tflite_commands(eval_dir, tflite_model, inference_type):
  return [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(tflite_model),
          '--batch_xs={}'.format(os.path.join(eval_dir, 'batch_xs.npy')),
          '--batch_ys={}'.format(os.path.join(eval_dir, 'output_ys.npy')),
          '--inference_type={}'.format(inference_type)]


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.tflite_model:
    raise ValueError('You must supply the frozen pb with --tflite_model')
  if FLAGS.inference_type != 'float' and FLAGS.inference_type != 'uint8':
    raise ValueError('--inference_type must be one of float or uint8')

  tf.logging.set_verbosity(tf.logging.INFO)
  tfrecord_pattern = [os.path.join(FLAGS.dataset_dir, 'validation-{:05d}-of-00128'.format(i)) for i in range(0, 128)]
  tf.logging.info('Import Dataset from tfrecord {}'.format(tfrecord_pattern))

  if FLAGS.max_num_batches:
    num_batches = FLAGS.max_num_batches
  else:
    num_records = sum([len(list(tf.python_io.tf_record_iterator(record))) for record in tfrecord_pattern])
    #num_records = len(list(tf.python_io.tf_record_iterator(tfrecord_pattern)))
    num_batches = int(math.ceil(num_records / float(FLAGS.batch_size)))

  #  for example in tf.python_io.tf_record_iterator(tfrecord_pattern):
	#  result = tf.train.Example.FromString(example)
	#  print(result)
	#  break

  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = prepare_imagenet_dataset(filenames, FLAGS.inference_type)
  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()

  tf.logging.info('Prepare run_tflite')
  eval_dir = os.path.dirname(FLAGS.tflite_model)
  eval_dir = os.path.join(eval_dir, 'eval_tflite')
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  cmds = prepare_run_tflite_commands(eval_dir,
                                     FLAGS.tflite_model, FLAGS.inference_type)

  tf.logging.info('Prepare metrics')
  with tf.name_scope("metrics"):
    lbls = tf.placeholder(tf.int32, [None, 1])
    if FLAGS.inference_type == 'float':
      preds = tf.placeholder(tf.int32, [None, 1001])
    elif FLAGS.inference_type == 'uint8':
      preds = tf.placeholder(tf.uint8, [None, 1001])
    accuracy, acc_update_op = tf.metrics.accuracy(lbls, tf.argmax(preds, 1))
    tf.summary.scalar('accuracy', accuracy)

  # Initialize `iterator` with dataset.
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer, feed_dict={filenames: tfrecord_pattern})

    for step in range(num_batches):
      if (step % 1000) == 0:
        print('{}/{}'.format(step, num_batches))
        #print(' '.join(cmds))
        print('  Accuracy: [{:.4f}]'.format(sess.run(accuracy)))
      images, labels = sess.run(next_batch)

      print(step)
      np.save(os.path.join(eval_dir, 'batch_xs.npy'), images)
      subprocess.check_output(cmds)
      ys = np.load(os.path.join(eval_dir, 'output_ys.npy'))
      sess.run(acc_update_op, feed_dict={lbls: labels, preds: ys})

    print('Accuracy: [{:.4f}]'.format(sess.run(accuracy)))


if __name__ == '__main__':
  tf.app.run()
