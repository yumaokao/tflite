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
    print(tf.argmax(output).eval())

  with tf.Graph().as_default() as graph_quant:
    tf.logging.info('Import frozen_quantor_pb model')
    tf.import_graph_def(graph_def_quant, name='')

  with tf.Session(graph=graph_quant) as sess:
    tf.logging.info('Process frozen_quantor_pb model')
    x = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))
    output_quant = sess.run(y, feed_dict={x: image_float})
    output_quant = np.squeeze(output_quant, 0)
    print(tf.argmax(output_quant).eval())

  # visualize
  tf.logging.info('Visualize the data')
  _, ax = plt.subplots(2, 2, sharex='col')

  image_float_flat = image_float.flatten()
  draw_distribution(ax[0, 0], image_float_flat, 'red')
  draw_distribution(ax[0, 1], output, 'green')
  draw_distribution(ax[1, 0], image_float_flat, 'red')
  draw_distribution(ax[1, 1], output_quant, 'green')

  plt.show()

if __name__ == '__main__':
  tf.app.run()
