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
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from prepare import prepare_dataset, prepare_metrics, prepare_tfrecords

'''
This program is used to visualize the result of different
models (float/fake-quant/quantized) given the same input data
'''

tf.app.flags.DEFINE_string(
    'frozen_pb', None, 'The GraphDef file of the freeze_graph.')
tf.app.flags.DEFINE_string(
    'frozen_quantor_pb', None, 'The GraphDef file of the freeze_graph with fake quantization.')
tf.app.flags.DEFINE_string(
    'tflite_model', None, 'The TFLite file of the tensorflow lite uint8 model.')
tf.app.flags.DEFINE_string(
    'input_node_name', 'input', 'The name of the input node.')
tf.app.flags.DEFINE_string(
    'input_npy_file', None, 'The numpy file of the float input data.')
tf.app.flags.DEFINE_string(
    'output_node_name', None, 'The name of the output node for visualization.')
tf.app.flags.DEFINE_string(
    'tensorflow_dir', None, 'The directory where the tensorflow are stored')
tf.app.flags.DEFINE_boolean(
    'dump_data', False, 'Whether to dump the input and output data for each batch or not.')

FLAGS = tf.app.flags.FLAGS

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

def draw_distribution(ax, data, fc):
  ax.hist(data, bins=100, edgecolor='gray', facecolor=fc, alpha=0.5, density=True)
  density = gaussian_kde(data)
  density.covariance_factor = lambda : .05
  density._compute_covariance()
  xs = np.linspace(np.min(data), np.max(data), 200)
  ax.plot(xs, density(xs), color='black', linewidth=2)
  ax.set_ylabel('density')

def get_tflite_quantization_info(model):
  # TODO: More straightforward solution?
  cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite',
          '{}'.format(model)]
  out = subprocess.check_output(cmd)
  for line in out.splitlines():
    if FLAGS.output_node_name + ' type' in line:
      result = re.search('quantization \((?P<scale>[0-9\.]+) (?P<zero>[0-9\.]+)\)', line)
      return float(result.group('scale')), int(result.group('zero'))
  raise ValueError('Quantization of the output node is not embedded inside the TFLite model')

def run_tflite(model, input_float):
  tmp_input_fn = 'input_tmp.npy'
  tmp_output_fn = 'output_tmp.npy'
  img_float = tf.convert_to_tensor(input_float)
  img_float = tf.multiply(img_float, 0.5)
  img_float = tf.add(img_float, 0.5)
  img_uint8 = tf.image.convert_image_dtype(img_float, dtype=tf.uint8)

  input_uint8 = img_uint8.eval()
  np.save(tmp_input_fn, input_uint8);
  cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(model),
          '--batch_xs={}'.format(tmp_input_fn),
          '--batch_ys={}'.format(tmp_output_fn),
          '--inference_type=uint8']
  subprocess.check_output(cmd)
  output_uint8 = np.load(tmp_output_fn)
  scale, zero_point = get_tflite_quantization_info(model)
  output_float = output_uint8.astype(float, copy=False)
  output_float = (output_float - zero_point) * scale

  subprocess.check_output(['rm', tmp_input_fn])
  subprocess.check_output(['rm', tmp_output_fn])
  return output_float

def main(_):
  if not FLAGS.frozen_pb:
    raise ValueError('--frozen_pb flag is required')
  if not FLAGS.input_npy_file:
    raise ValueError('--input_npy_file flag is required')

  tf.logging.set_verbosity(tf.logging.INFO)

  tf.logging.info('Load GraphDef from frozen_pb {}'.format(FLAGS.frozen_pb))
  graph_def = load_graph_def(FLAGS.frozen_pb)
  graph_def_quant = load_graph_def(FLAGS.frozen_quantor_pb)
  image_float = np.load(FLAGS.input_npy_file)

  with tf.Graph().as_default() as graph_float:
    tf.logging.info('Import frozen_pb model')
    tf.import_graph_def(graph_def, name='')

  with tf.Session(graph=graph_float) as sess:
    tf.logging.info('Process frozen_pb model')
    x = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))
    output = sess.run(y, feed_dict={x: image_float})
    output = np.squeeze(output, 0)
    #print(tf.argmax(output).eval())

  with tf.Graph().as_default() as graph_quant:
    tf.logging.info('Import frozen_quantor_pb model')
    tf.import_graph_def(graph_def_quant, name='')

  with tf.Session(graph=graph_quant) as sess:
    tf.logging.info('Process frozen_quantor_pb model')
    x = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))
    output_quant = sess.run(y, feed_dict={x: image_float})
    output_quant = np.squeeze(output_quant, 0)
    #print(tf.argmax(output_quant).eval())

  with tf.Session(graph=graph_quant) as sess:
    tf.logging.info('Process tflite model')
    output_lite = run_tflite(FLAGS.tflite_model, image_float)
    output_lite = np.squeeze(output_lite, 0)
    #print(tf.argmax(output_lite).eval())

  # visualize
  tf.logging.info('Visualize the data')
  _, ax = plt.subplots(3, 2, sharex='col')

  image_float_flat = image_float.flatten()
  draw_distribution(ax[0, 0], image_float_flat, 'red')
  draw_distribution(ax[0, 1], output, 'green')
  draw_distribution(ax[1, 0], image_float_flat, 'red')
  draw_distribution(ax[1, 1], output_quant, 'green')
  draw_distribution(ax[2, 0], image_float_flat, 'red')
  draw_distribution(ax[2, 1], output_lite, 'green')

  plt.show()

if __name__ == '__main__':
  tf.app.run()
