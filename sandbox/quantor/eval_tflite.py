from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tempfile
import subprocess
import numpy as np
import tensorflow as tf

from prepare import prepare_dataset, prepare_metrics, prepare_tfrecords


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
  tfrecords = prepare_tfrecords(FLAGS.dataset_name, FLAGS.dataset_dir,
                                FLAGS.dataset_split_name)

  if FLAGS.max_num_batches:
    num_batches = FLAGS.max_num_batches
  else:
    num_records = sum([len(list(tf.python_io.tf_record_iterator(r)))
                       for r in tfrecords])
    num_batches = int(math.ceil(num_records / float(FLAGS.batch_size)))

  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = prepare_dataset(filenames, FLAGS.dataset_name, FLAGS.input_size,
                            batch_size=FLAGS.batch_size,
                            inference_type=FLAGS.inference_type)
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
  lbls, preds, accuracy, acc_update_op = prepare_metrics(
          FLAGS.dataset_name, inference_type=FLAGS.inference_type)

  # Initialize `iterator` with dataset.
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer, feed_dict={filenames: tfrecords})

    for step in range(num_batches):
      if (step % 1000) == 0:
        print('{}/{}'.format(step, num_batches))
        # print(' '.join(cmds))
        print('  Accuracy: [{:.4f}]'.format(sess.run(accuracy)))
      images, labels = sess.run(next_batch)

      np.save(os.path.join(eval_dir, 'batch_xs.npy'), images)
      subprocess.check_output(cmds)
      ys = np.load(os.path.join(eval_dir, 'output_ys.npy'))
      sess.run(acc_update_op, feed_dict={lbls: labels, preds: ys})

    print('Accuracy: [{:.4f}]'.format(sess.run(accuracy)))


if __name__ == '__main__':
  tf.app.run()
