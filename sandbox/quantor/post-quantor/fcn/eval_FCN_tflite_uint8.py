from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import subprocess
import re

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
from tensorflow.python.framework import graph_util

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for evaluation")
tf.flags.DEFINE_integer("num_batches", "10000", "number of batch for evaluation")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string(
    'tflite_model', None, 'The input quantized TFLite model file.')
tf.flags.DEFINE_string(
    'tensorflow_dir', None, 'The directory where the tensorflow are stored')
# This should be parsed from the model, FIXME
tf.flags.DEFINE_string('output_node_name', None, 'The name of the output node of the tflite model.')

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSES = 151
IMAGE_SIZE = 224

def prepare_run_tflite_commands(eval_dir):
  return [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(FLAGS.tflite_model),
          '--batch_xs={}'.format(os.path.join(eval_dir, 'batch_xs.npy')),
          '--batch_ys={}'.format(os.path.join(eval_dir, 'output_ys.npy')),
          '--inference_type=uint8']

def get_tflite_quantization_info():
  cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite', FLAGS.tflite_model]
  out = subprocess.check_output(cmd)
  for line in out.splitlines():
    if FLAGS.output_node_name + ' type' in line:
      result = re.search('quantization \((?P<scale>[0-9\.]+) (?P<zero>[0-9\.]+)\)', line)
      return float(result.group('scale')), int(result.group('zero'))
  raise ValueError('Quantization of the output node is not embedded inside the TFLite model')

def main(argv=None):
    if not FLAGS.tensorflow_dir:
      raise ValueError('You must supply the tensorflow directory with --tensorflow_dir')
    if not FLAGS.tflite_model:
      raise ValueError('You must supply the frozen pb with --tflite_model')
    if not FLAGS.output_node_name:
      raise ValueError('Please specify the output node name (with --output_node_name flag)')

    logits = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES], name="logits")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))

    print("Preparing run_tflite command...")
    eval_dir = os.path.dirname(FLAGS.tflite_model)
    cmds = prepare_run_tflite_commands(eval_dir)

    print("Setting up image reader...")
    _, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(valid_records))

    print("Setting up dataset reader...")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    total_loss = 0
    for itr in xrange(FLAGS.num_batches):
        valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
        np.save(os.path.join(eval_dir, 'origin.npy'), valid_images)
        np.save(os.path.join(eval_dir, 'batch_xs.npy'), valid_images)
        #print(' '.join(cmds))
        subprocess.check_output(cmds)
        ys = np.load(os.path.join(eval_dir, 'output_ys.npy'))
        scale, zero_point = get_tflite_quantization_info()
        ys = (ys.astype(np.float32) - zero_point) * scale
        np.save(os.path.join(eval_dir, 'annotation.npy'), valid_annotations)

        with tf.Session() as sess:
            valid_loss = sess.run(loss, feed_dict={logits: ys, annotation: valid_annotations})
            total_loss += valid_loss
            print('%d iteration: validation loss = %g' % (itr+1, valid_loss))

    sess.close()


if __name__ == "__main__":
    tf.app.run()
