from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import subprocess
import shutil
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer(
    'batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'num_batches', None,
    'number of batches to evaluate.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the jpeg image files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', None,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'preprocess_name', None, 'The name of the preprocessing method '
    'either inception [-1.0, 1.0] or vgg [-105.0, 141.0]')
tf.app.flags.DEFINE_string(
    'tflite_model', None, 'The TFLite model.')
tf.app.flags.DEFINE_string(
    'input_node_name', None, 'The name of the input node.')
tf.app.flags.DEFINE_string(
    'output_node_name', None, 'The name of the output node.')
tf.app.flags.DEFINE_integer(
    'input_size', None, 'The width/height of the input image.')
tf.app.flags.DEFINE_integer(
    'log_step', 100, 'The log step size (# batches)')
tf.app.flags.DEFINE_string(
    'inference_type', None, 'The inference type (float or uint8)')
tf.app.flags.DEFINE_string(
    'tensorflow_dir', None, 'The TensorFlow directory')

FLAGS = tf.app.flags.FLAGS
GOLDEN_COUNT = 5

def tflite_preprocess(fns):

  image_fns = map(lambda f: f + '.jpeg', fns)
  label_fns = map(lambda f: f + '.log', fns)

  # Read the label file
  fn_queue = tf.train.string_input_producer(label_fns, shuffle=False, num_epochs=1)
  reader = tf.WholeFileReader()
  _, content = reader.read(fn_queue)
  label = tf.string_to_number(tf.string_split([content], delimiter='\n').values[0], out_type=tf.int32)
  label -= FLAGS.labels_offset

  # Read and decode jpeg
  fn_queue = tf.train.string_input_producer(image_fns, shuffle=False, num_epochs=1)
  reader = tf.WholeFileReader()
  _, content = reader.read(fn_queue)
  image = tf.image.decode_jpeg(content, channels=3)

  if FLAGS.preprocess_name == 'inception':
    if FLAGS.inference_type == 'float':
      image = tf.image.convert_image_dtype(image, dtype=tf.float32) # value range [0.0, 1.0]
      image = tf.image.central_crop(image, central_fraction=0.875)
      image = tf.expand_dims(image, 0) # resize_bilinear requires 4D shape
      image = tf.image.resize_bilinear(image, [FLAGS.input_size, FLAGS.input_size])
      image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5) # value range [-0.5, 0.5]
      image = tf.multiply(image, 2.0) # value range [-1.0, 1.0]
    elif FLAGS.inference_type == 'uint8':
      image = tf.image.convert_image_dtype(image, dtype=tf.float32) # value range [0.0, 1.0]
      image = tf.image.central_crop(image, central_fraction=0.875)
      image = tf.expand_dims(image, 0) # resize_bilinear requires 4D shape
      image = tf.image.resize_bilinear(image, [FLAGS.input_size, FLAGS.input_size])
      image = tf.squeeze(image, [0])
      image = tf.image.convert_image_dtype(image, dtype=tf.uint8) # value range [0, 255]

  # Make batches
  label, image = tf.train.batch([label, image], FLAGS.batch_size)
  return label, image

def prepare_run_tflite_commands(eval_dir, tflite_model, inference_type):
  return [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(tflite_model),
          '--batch_xs={}'.format(os.path.join(eval_dir, 'input.npy')),
          '--batch_ys={}'.format(os.path.join(eval_dir, 'output.npy')),
          '--inference_type={}'.format(inference_type)]

def main(_):
  if FLAGS.dataset_dir is None:
    raise ValueError('Please specify --dataset_dir')
  if FLAGS.batch_size is None:
    raise ValueError('Please specify --batch_size')
  if FLAGS.num_batches is None:
    raise ValueError('Please specify --num_batches')
  if FLAGS.labels_offset is None:
    raise ValueError('Please specify --labels_offset')
  if FLAGS.preprocess_name is None:
    raise ValueError('Please specify --preprocess_name')
  if FLAGS.tflite_model is None:
    raise ValueError('Please specify --tflite_model')
  if FLAGS.input_node_name is None:
    raise ValueError('Please specify --input_node_name')
  if FLAGS.output_node_name is None:
    raise ValueError('Please specify --output_node_name')
  if FLAGS.input_size is None:
    raise ValueError('Please specify --input_size')
  if FLAGS.inference_type is None:
    raise ValueError('Please specify --inference_type')
  if FLAGS.tensorflow_dir is None:
    raise ValueError('Please specify --tensorflow_dir')
  if FLAGS.inference_type not in ['float', 'uint8']:
    raise ValueError('--inference_type should be either \'float\' or \'uint8\'')

  # Parse dataset_dir
  files = [os.path.join(FLAGS.dataset_dir, os.path.splitext(f)[0])
              for f in os.listdir(FLAGS.dataset_dir) if os.path.isfile(os.path.join(FLAGS.dataset_dir, f))]
  files = list(set(files))
  files = [os.path.abspath(f) for f in files]
  random.shuffle(files)

  # Preprocess
  next_label, next_batch = tflite_preprocess(files)

  # Directory for all the feature map and labels
  dump_dir = os.path.dirname(FLAGS.tflite_model)
  dump_dir = os.path.join(dump_dir, 'dump_tflite_jpeg')
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

  # Directory for the golden input/output pair
  eval_dir = os.path.join(dump_dir, 'golden')
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

  # Directory for the tmp data
  tmp_dir = os.path.join(dump_dir, 'tmp')
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

  cmds = prepare_run_tflite_commands(tmp_dir, FLAGS.tflite_model, FLAGS.inference_type)

  with tf.Session() as sess:

    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Start the queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(FLAGS.num_batches):

      if i != 0 and i % FLAGS.log_step == 0:
        print('Finish processing {} batches ...'.format(i))

      label, image = sess.run([next_label, next_batch])

      np.save(os.path.join(tmp_dir, 'input.npy'), image)
      subprocess.check_output(cmds)
      pred = np.load(os.path.join(tmp_dir, 'output.npy'))

      if i < GOLDEN_COUNT:
        np.save(os.path.join(eval_dir, 'input_{}.npy'.format(i)), image)
        np.save(os.path.join(eval_dir, 'output_{}.npy'.format(i)), pred)

      np.save(os.path.join(dump_dir, '{}_pred.npy'.format(i)), pred)
      np.save(os.path.join(dump_dir, '{}_label.npy'.format(i)), label)

    print('Finish processing ALL batches')

    # Stop the queue
    coord.request_stop()
    coord.join(threads)

  # Remove tmp directory
  shutil.rmtree(eval_dir)


if __name__ == '__main__':
  tf.app.run()
