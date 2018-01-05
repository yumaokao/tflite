from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf

import tensorflow.contrib.quantize as qg

tf.app.flags.DEFINE_string(
    'input_graph', None, 'The GraphDef file are stored with freeze_graph.')

tf.app.flags.DEFINE_string(
    'output_graph', None, 'The GraphDef file are stored with freeze_graph.')

FLAGS = tf.app.flags.FLAGS


def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def main(_):
  if not FLAGS.input_graph:
    raise ValueError('You must supply the frozen pb with --input_graph')
  if not FLAGS.output_graph:
    raise ValueError('You must supply the frozen pb with --input_graph')

  out_dirname, out_basename = os.path.split(FLAGS.output_graph)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Load GraphDef from input_graph {}'.format(FLAGS.input_graph))
  graph_def = load_graph_def(FLAGS.input_graph)

  # Initialize `iterator` with training data.
  with tf.Session() as sess:
    tf.import_graph_def(graph_def, name='')
    quantized_graph = qg.create_training_graph(sess.graph)
    tf.train.write_graph(quantized_graph, out_dirname,
                         'quantized-{}'.format(out_basename), as_text=False)

  with tf.Session(graph=quantized_graph) as sess:
    import ipdb
    ipdb.set_trace()
    print(sess.graph)


if __name__ == '__main__':
  tf.app.run()
