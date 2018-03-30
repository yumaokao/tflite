from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import struct
import os

def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

tf.app.flags.DEFINE_string(
    'frozen_pb', None, 'The input vgg graph.')
tf.app.flags.DEFINE_string(
    'output_pb', None, 'The output transformed graph.')

FLAGS = tf.app.flags.FLAGS

def main(_):
  graph_def = load_graph_def(FLAGS.frozen_pb)
  with tf.Session() as sess:
    tf.import_graph_def(graph_def, name='')
    graph = sess.graph

    prev_tensor = graph.get_tensor_by_name('vgg_16/pool5/MaxPool:0')
    batch_num_tensor = tf.shape(prev_tensor, name='vgg_16/fc6/fc/shape')
    batch_num_tensor = tf.gather(batch_num_tensor, 0, name='vgg_16/fc6/fc/gather')
    prev_tensor = tf.reshape(prev_tensor, [batch_num_tensor, -1], name='vgg_16/fc6/fc/reshape')
    weight_tensor = graph.get_tensor_by_name('vgg_16/fc6/weights/read:0')
    weight_tensor = tf.reshape(weight_tensor, [-1, weight_tensor.get_shape()[-1]], name='vgg_16/fc6/fc/weights/reshape')
    bias_tensor = graph.get_tensor_by_name('vgg_16/fc6/biases/read:0')
    matmul_tensor = tf.matmul(prev_tensor, weight_tensor, name='vgg_16/fc6/fc/MatMul')
    matmul_tensor = tf.nn.bias_add(matmul_tensor, bias_tensor, data_format='NHWC', name='vgg_16/fc6/fc/BiasAdd')
    matmul_tensor = tf.nn.relu(matmul_tensor, name='vgg_16/fc6/fc/Relu')

    prev_tensor = matmul_tensor
    weight_tensor = graph.get_tensor_by_name('vgg_16/fc7/weights/read:0')
    weight_tensor = tf.reshape(weight_tensor, [-1, weight_tensor.get_shape()[-1]], name='vgg_16/fc7/fc/weights/reshape')
    bias_tensor = graph.get_tensor_by_name('vgg_16/fc7/biases/read:0')
    matmul_tensor = tf.matmul(prev_tensor, weight_tensor, name='vgg_16/fc7/fc/MatMul')
    matmul_tensor = tf.nn.bias_add(matmul_tensor, bias_tensor, data_format='NHWC', name='vgg_16/fc7/fc/BiasAdd')
    matmul_tensor = tf.nn.relu(matmul_tensor, name='vgg_16/fc7/fc/Relu')

    prev_tensor = matmul_tensor
    weight_tensor = graph.get_tensor_by_name('vgg_16/fc8/weights/read:0')
    weight_tensor = tf.reshape(weight_tensor, [-1, weight_tensor.get_shape()[-1]], name='vgg_16/fc8/fc/weights/reshape')
    bias_tensor = graph.get_tensor_by_name('vgg_16/fc8/biases/read:0')
    matmul_tensor = tf.matmul(prev_tensor, weight_tensor, name='vgg_16/fc8/fc/MatMul')
    matmul_tensor = tf.nn.bias_add(matmul_tensor, bias_tensor, data_format='NHWC', name='vgg_16/fc8/fc/BiasAdd')

    output_dir = os.path.dirname(FLAGS.output_pb)
    pb_name = os.path.basename(FLAGS.output_pb)
    tf.train.write_graph(graph, output_dir, pb_name, as_text=False)

if __name__ == '__main__':
  tf.app.run()
