from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
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
    'Max number of batches to evaluate (use all batches by default).')
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'summary_dir', None, 'The directory where summaries save.')
tf.app.flags.DEFINE_string(
    'frozen_pb', None, 'The GraphDef file of the freeze_graph.')
tf.app.flags.DEFINE_string(
    'input_node_name', 'input', 'The name of the input node.')
tf.app.flags.DEFINE_string(
    'output_node_name', None, 'The name of the output node.')
tf.app.flags.DEFINE_integer(
    'input_size', 299, 'The width/height of the input image.')
tf.app.flags.DEFINE_string(
    'toco_inference_type', 'float', 'The inference type to run toco')
tf.app.flags.DEFINE_string(
    'tensorflow_dir', None, 'The directory where the tensorflow are stored')
tf.app.flags.DEFINE_string(
    'evaluation_mode', 'statistics', 'The evaluation method.')
tf.app.flags.DEFINE_float(
    'evaluation_threshold', 0.01, 'The evaluation threshold (for "diff_threshold" evaluation_mode only).')
tf.app.flags.DEFINE_string(
    'extra_toco_flags', '', 'The extra command for toco tool.')
tf.app.flags.DEFINE_boolean(
    'dump_data', False, 'Whether to dump the input and output data for each batch or not.')

FLAGS = tf.app.flags.FLAGS
EVAL_MODE = ['statistics', 'diff_threshold', 'top1', 'accuracy']

def get_statistics(numpy_data):
  data = []
  data.append('Q0: {}'.format(np.percentile(numpy_data, 0)))
  data.append('Q1: {}'.format(np.percentile(numpy_data, 25)))
  data.append('Q2: {}'.format(np.percentile(numpy_data, 50)))
  data.append('Q3: {}'.format(np.percentile(numpy_data, 75)))
  data.append('Q4: {}'.format(np.percentile(numpy_data, 100)))
  return data

def get_tflite_quantization_info(work_dir):
  # TODO: More straightforward solution?
  cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite',
          '{}'.format(os.path.join(work_dir, '{}_model.lite'.format(FLAGS.toco_inference_type)))]
  out = subprocess.check_output(cmd)
  for line in out.splitlines():
    if FLAGS.output_node_name + ' type' in line:
      result = re.search('quantization \((?P<scale>[0-9\.]+) (?P<zero>[0-9\.]+)\)', line)
      return float(result.group('scale')), int(result.group('zero'))
  raise ValueError('Quantization of the output node is not embedded inside the TFLite model')

def process_toco(work_dir):
  # TODO: Fixed the bug in toco when transforming the batchnorm op
  # Currently we use the transform_graph tool in tensorflow to handle the batchnorm op
  pre_process_cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/tools/graph_transforms/transform_graph',
                      '--in_graph={}'.format(FLAGS.frozen_pb),
                      '--out_graph={}'.format(FLAGS.frozen_pb + '.tmp'),
                      '--inputs={}'.format(FLAGS.input_node_name),
                      '--outputs={}'.format(FLAGS.output_node_name),
                      '--transforms=remove_nodes(op=Identity, op=CheckNumerics) fold_batch_norms fold_old_batch_norms']
  subprocess.check_output(pre_process_cmd)

  # TODO: should the value of mean_values and std_values flag be passed from user?
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
		      '--dump_graphviz={}'.format(work_dir),
          '--mean_values=128',
          '--std_values=127']

  extra_flags = FLAGS.extra_toco_flags.split()
  for e_flag in extra_flags:
    if any([e_flag.split('=')[0] in flag for flag in cmd]):
      # TODO: remove the origin flag when conflicts happen
      tf.logging.warning('Warning: Extra toco flag "{}" has been set to "{}".'.format(e_flag.split('=')[0][2:], e_flag.split('=')[1]))
    else:
      cmd.append(e_flag)
  subprocess.check_output(cmd)

  rm_cmd = ['rm', FLAGS.frozen_pb + '.tmp']
  subprocess.check_output(rm_cmd)

def prepare_run_tflite_commands(work_dir, data_dir, input_fn, output_fn):
  return [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(os.path.join(work_dir, '{}_model.lite'.format(FLAGS.toco_inference_type))),
          '--batch_xs={}'.format(os.path.join(data_dir, input_fn)),
          '--batch_ys={}'.format(os.path.join(data_dir, output_fn)),
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
    raise ValueError('You must supply the frozen pb (with fake quantization) with --frozen_pb')
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
    tf.logging.info('Total batch number: {}'.format(num_batches))

  tf.logging.info('Prepare Dataset from tfrecord[0] '.format(tfrecords[0]))
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset_tf = prepare_dataset(filenames, FLAGS.dataset_name, FLAGS.input_size,
                            batch_size=FLAGS.batch_size, inference_type='float')
  iterator_tf = dataset_tf.make_initializable_iterator()
  next_batch_tf = iterator_tf.get_next()

  dataset_lite = prepare_dataset(filenames, FLAGS.dataset_name, FLAGS.input_size,
                            batch_size=FLAGS.batch_size, inference_type=FLAGS.toco_inference_type)
  iterator_lite = dataset_lite.make_initializable_iterator()
  next_batch_lite = iterator_lite.get_next()

  tf.logging.info('Load GraphDef from frozen_pb {}'.format(FLAGS.frozen_pb))
  graph_def = load_graph_def(FLAGS.frozen_pb)

  tf.logging.info('Run toco')
  toco_cmds = process_toco(work_dir)

  tf.logging.info('Prepare metrics')
  lbls_tf, preds_tf, accuracy_tf, acc_update_op_tf = prepare_metrics(FLAGS.dataset_name)
  lbls_lite, preds_lite, accuracy_lite, acc_update_op_lite = prepare_metrics(FLAGS.dataset_name)

  if FLAGS.summary_dir:
    tf.logging.info('Prepare summary writer')
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)
    summaries = tf.summary.merge_all()

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(iterator_tf.initializer, feed_dict={filenames: tfrecords})
    sess.run(iterator_lite.initializer, feed_dict={filenames: tfrecords})

    tf.import_graph_def(graph_def, name='')
    graph = sess.graph

    # get x and y
    x = graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))

    for step in range(num_batches):
      tf.logging.info('Processing batch #{} ...'.format(step))

      images_tf, labels_tf = sess.run(next_batch_tf)
      images_lite, labels_lite = sess.run(next_batch_lite)

      assert labels_tf == labels_lite

      if FLAGS.dump_data == True:
        input_tf_fn = 'tf_xs_{}.npy'.format(step);
        input_lite_fn = 'lite_xs_{}.npy'.format(step);
        output_tf_fn = 'tf_ys_{}.npy'.format(step);
        output_lite_fn = 'lite_ys_{}.npy'.format(step);
      else:
        input_tf_fn = 'tf_xs.npy'
        input_lite_fn = 'lite_xs.npy'
        output_tf_fn = 'tf_ys.npy'
        output_lite_fn = 'lite_ys.npy'

      # forward GraphDef
      np.save(os.path.join(data_dir, input_tf_fn), images_tf)
      ys_tf = sess.run(y, feed_dict={x: images_tf})
      np.save(os.path.join(data_dir, output_tf_fn), ys_tf)

      # forward TFLite
      np.save(os.path.join(data_dir, input_lite_fn), images_lite)
      tflite_cmds = prepare_run_tflite_commands(work_dir, data_dir, input_lite_fn, output_lite_fn)
      subprocess.check_output(tflite_cmds)
      ys_lite = np.load(os.path.join(data_dir, output_lite_fn))

      # convert the TFLite result back to float
      if FLAGS.toco_inference_type == 'uint8':
        scale, zero_point = get_tflite_quantization_info(work_dir)
        ys_lite = ys_lite.astype(float, copy=False)
        ys_lite = (ys_lite - zero_point) * scale

      # Evaluate the result
      if FLAGS.evaluation_mode == 'statistics':
        tf.logging.info('=== Tensorflow output statistics for batch #{} ==='.format(step))
        tf.logging.info(' {}'.format(get_statistics(ys_tf)))
        tf.logging.info('=== TFLite output statistics for batch #{} ==='.format(step))
        tf.logging.info(' {}'.format(get_statistics(ys_lite)))
      elif FLAGS.evaluation_mode == 'diff_threshold':
        abs_diff = np.fabs(ys_tf - ys_lite)
        count = abs_diff > FLAGS.evaluation_threshold # True if it exceeds the threshold
        if (count.sum() == 0):
          tf.logging.info('=== PASSED for batch #{} ==='.format(step))
          tf.logging.info(' All {} output elements pass the threshold {}.'.format(
                abs_diff.size, FLAGS.evaluation_threshold))
        else:
          tf.logging.info('=== FAILED for batch #{} ==='.format(step))
          tf.logging.info(' {} of the total {} output elements exceed the threshold {}.'.format(
                count.sum(), abs_diff.size, FLAGS.evaluation_threshold))
      elif FLAGS.evaluation_mode == 'top1':
        tf_result = tf.argmax(ys_tf, 1).eval()
        lite_result = tf.argmax(ys_lite, 1).eval()
        labels = np.reshape(labels_tf, [FLAGS.batch_size])
        tf.logging.info('=== For batch #{} ==='.format(step))
        tf.logging.info(' Correct label:     {}'.format(labels))
        tf.logging.info(' Tensorflow result: {}'.format(tf_result))
        tf.logging.info(' TFLite result:     {}'.format(lite_result))
      elif FLAGS.evaluation_mode == 'accuracy':
        sess.run(acc_update_op_tf, feed_dict={lbls_tf: labels_tf, preds_tf: ys_tf})
        sess.run(acc_update_op_lite, feed_dict={lbls_lite: labels_tf, preds_lite: ys_lite})
        if ((step + 1) % 50) == 0:
          # Print current accuracy
          tf.logging.info('=== Current accuracy for {}/{} batches ==='.format(step + 1, num_batches))
          tf.logging.info(' Tensorflow: {}'.format(accuracy_tf.eval()))
          tf.logging.info(' TFLite:     {}'.format(accuracy_lite.eval()))

    if FLAGS.evaluation_mode == 'accuracy':
      tf.logging.info('=== Final accuracy for {} {} ==='.format(num_batches, 'batch' if num_batches == 1 else 'batches'))
      tf.logging.info(' Tensorflow: {}'.format(accuracy_tf.eval()))
      tf.logging.info(' TFLite:     {}'.format(accuracy_lite.eval()))

    if FLAGS.summary_dir:
      summary_writer.add_graph(sess.graph)


if __name__ == '__main__':
  tf.app.run()
