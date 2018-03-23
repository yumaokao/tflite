from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import subprocess

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
from tensorflow.python.framework import graph_util

VGG_MEAN = [123.680, 116.779, 103.939]

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for evaluation")
tf.flags.DEFINE_integer("num_batches", "10000", "number of batch for evaluation")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string(
    'tflite_model', None, 'The TFLite model file is stored with toco.')
tf.flags.DEFINE_string(
    'inference_type', 'float', 'The inference type to run the tflie model')
tf.flags.DEFINE_string(
    'tensorflow_dir', None, 'The directory where the tensorflow are stored')

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSES = 151
IMAGE_SIZE = 224

def prepare_run_tflite_commands(eval_dir):
  return [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(FLAGS.tflite_model),
          '--batch_xs={}'.format(os.path.join(eval_dir, 'batch_xs.npy')),
          '--batch_ys={}'.format(os.path.join(eval_dir, 'output_ys.npy')),
          '--inference_type={}'.format(FLAGS.inference_type)]

def main(argv=None):
    if not FLAGS.tensorflow_dir:
        raise ValueError('You must supply the tensorflow directory with --tensorflow_dir')
    if not FLAGS.tflite_model:
        raise ValueError('You must supply the frozen pb with --tflite_model')
    if FLAGS.inference_type != 'float' and FLAGS.inference_type != 'uint8':
        raise ValueError('--inference_type must be one of float or uint8')

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
        valid_images = valid_images - np.array(VGG_MEAN)
        valid_images = valid_images.astype(np.float32)
        np.save(os.path.join(eval_dir, 'batch_xs.npy'), valid_images)
        subprocess.check_output(cmds)
        ys = np.load(os.path.join(eval_dir, 'output_ys.npy'))

        with tf.Session() as sess:
          valid_loss = sess.run(loss, feed_dict={logits: ys, annotation: valid_annotations})
          total_loss += valid_loss

        if itr != 0 and itr % 10 == 0:
            print('%d iteration: Average validation loss = %g' % (itr, total_loss / itr))

    sess.close()


if __name__ == "__main__":
    tf.app.run()
