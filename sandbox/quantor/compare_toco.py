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

'''
This program is used to compare the output of GraphDef
and TFLite model based on the given input tensor
'''

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
    'summary_dir', None, 'The directory where summaries save.')
tf.app.flags.DEFINE_string(
    'frozen_pb', None, 'The GraphDef file are stored with freeze_graph.')
tf.app.flags.DEFINE_string(
    'input_node_name', 'input', 'The name of the input node.')
tf.app.flags.DEFINE_string(
    'output_node_name', None, 'The name of the output node.')
tf.app.flags.DEFINE_integer(
    'input_size', 299, 'The width/height of the input image.')
tf.app.flags.DEFINE_string(
    'toco_inference_type', 'float', 'The inference type to run the tflite model')
tf.app.flags.DEFINE_string(
    'tensorflow_dir', None, 'The directory where the tensorflow are stored')
tf.app.flags.DEFINE_string(
    'evaluation_mode', 'statistics', 'The evaluation method.')
tf.app.flags.DEFINE_float(
    'evaluation_threshold', 0.01, 'The evaluation threshold (for "diff_threshold" mode).')

FLAGS = tf.app.flags.FLAGS
EVAL_MODE = ['statistics', 'diff_threshold']

def get_statistics(numpy_data):
  data = []
  data.append('Q0: {}'.format(np.percentile(numpy_data, 0)))
  data.append('Q1: {}'.format(np.percentile(numpy_data, 25)))
  data.append('Q2: {}'.format(np.percentile(numpy_data, 50)))
  data.append('Q3: {}'.format(np.percentile(numpy_data, 75)))
  data.append('Q4: {}'.format(np.percentile(numpy_data, 100)))
  return data

def prepare_toco_commands(work_dir):
  # TODO: Fixed the bug in toco when transforming the batchnorm op
  # Currently we use the transform_graph tool in tensorflow to handle the batchnorm op
  pre_process_cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/tools/graph_transforms/transform_graph',
                      '--in_graph={}'.format(FLAGS.frozen_pb),
                      '--out_graph={}'.format(FLAGS.frozen_pb + '.tmp'),
                      '--inputs={}'.format(FLAGS.input_node_name),
                      '--outputs={}'.format(FLAGS.output_node_name),
                      '--transforms=remove_nodes(op=Identity, op=CheckNumerics) fold_batch_norms fold_old_batch_norms']
  subprocess.check_output(pre_process_cmd)

  cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/toco/toco',
          '--input_file={}'.format(FLAGS.frozen_pb + '.tmp'),
		      '--input_format=TENSORFLOW_GRAPHDEF',
          '--output_format=TFLITE',
		      '--output_file={}'.format(os.path.join(work_dir, '{}_model.lite'.format(FLAGS.toco_inference_type))),
          '--inference_type={}'.format('FLOAT' if FLAGS.toco_inference_type == 'float' else 'QUANTIZED_UINT8'),
          '--inference_input_type={}'.format('FLOAT' if FLAGS.toco_inference_type == 'float' else 'QUANTIZED_UINT8'),
          '--input_arrays={}'.format(FLAGS.input_node_name),
		      '--output_arrays={}'.format(FLAGS.output_node_name),
          '--input_shapes=1,{0},{0},3'.format(FLAGS.input_size),
		      '--dump_graphviz={}'.format(work_dir)]
  return cmd

def prepare_run_tflite_commands(work_dir, data_dir):
  return [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(os.path.join(work_dir, '{}_model.lite'.format(FLAGS.toco_inference_type))),
          '--batch_xs={}'.format(os.path.join(data_dir, 'batch_xs.npy')),
          '--batch_ys={}'.format(os.path.join(data_dir, 'tflite_ys.npy')),
          '--inference_type={}'.format(FLAGS.toco_inference_type)]

def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    '''
    for node in graph_def.node:
      if (len(node.input) > 0 and node.op != 'Identity'):
        print(node.name + '  ' + node.op)
    '''
  return graph_def

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.frozen_pb:
    raise ValueError('You must supply the frozen pb with --frozen_pb')
  if not FLAGS.output_node_name:
    raise ValueError('You must supply the output node name with --output_node_name')
  if not FLAGS.tensorflow_dir:
    raise ValueError('You must supply the tensorflow path with --tensorflow_dir')
  if not FLAGS.evaluation_mode in EVAL_MODE:
    raise ValueError('The value of --evaluation_mode should be one of the following: {}'.format(', '.join(EVAL_MODE)))
  if FLAGS.toco_inference_type != 'float' and FLAGS.toco_inference_type != 'uint8':
    raise ValueError('--toco_inference_type must be one of float or uint8')

  work_dir = os.path.join(os.path.dirname(FLAGS.frozen_pb), 'compare_toco')
  if not os.path.exists(work_dir):
    os.makedirs(work_dir)
  data_dir = os.path.join(work_dir, 'data')
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  tf.logging.set_verbosity(tf.logging.INFO)
  tfrecords = prepare_tfrecords(FLAGS.dataset_name, FLAGS.dataset_dir,
                                FLAGS.dataset_split_name)

  if FLAGS.max_num_batches:
    num_batches = FLAGS.max_num_batches
  else:
    num_records = sum([len(list(tf.python_io.tf_record_iterator(r)))
                       for r in tfrecords])
    num_batches = int(math.ceil(num_records / float(FLAGS.batch_size)))

  tf.logging.info('Prepare Dataset from tfrecord[0] '.format(tfrecords[0]))
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = prepare_dataset(filenames, FLAGS.dataset_name, FLAGS.input_size,
                            batch_size=FLAGS.batch_size)
  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()

  tf.logging.info('Load GraphDef from frozen_pb {}'.format(FLAGS.frozen_pb))
  graph_def = load_graph_def(FLAGS.frozen_pb)

  tf.logging.info('Run toco')
  toco_cmds = prepare_toco_commands(work_dir)
  subprocess.check_output(toco_cmds)

  tf.logging.info('Prepare tflite command')
  tflite_cmds = prepare_run_tflite_commands(work_dir, data_dir)

  tf.logging.info('Prepare metrics')
  lbls, preds, accuracy, acc_update_op = prepare_metrics(FLAGS.dataset_name)

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
      images, labels = sess.run(next_batch)

      # forward GraphDef
      ys_tf = sess.run(y, feed_dict={x: images})
      np.save(os.path.join(data_dir, 'tf_ys.npy'), ys_tf)

      # forward TFLite
      np.save(os.path.join(data_dir, 'batch_xs.npy'), images)
      subprocess.check_output(tflite_cmds)
      ys_lite = np.load(os.path.join(data_dir, 'tflite_ys.npy'))

      # Evaluate the result
      if FLAGS.evaluation_mode == 'statistics':
        print('=== GraphDef output statistics ===')
        print(get_statistics(ys_tf))
        print('=== TFLite output statistics ===')
        print(get_statistics(ys_lite))
      elif FLAGS.evaluation_mode == 'diff_threshold':
        abs_diff = np.fabs(ys_tf - ys_lite)
        count = abs_diff > FLAGS.evaluation_threshold # True if it exceeds the threshold
        if (count.sum() == 0):
          print('=== PASSED ===')
          print('All {} output elements pass the threshold {}.'.format(
                abs_diff.size, FLAGS.evaluation_threshold))
        else:
          print('=== FAILED ===')
          print('{} of the total {} output elements exceed the threshold {}.'.format(
                count.sum(), abs_diff.size, FLAGS.evaluation_threshold))

    if FLAGS.summary_dir:
      summary_writer.add_graph(sess.graph)


if __name__ == '__main__':
  tf.app.run()
