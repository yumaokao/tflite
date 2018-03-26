from __future__ import print_function
import tensorflow as tf
import numpy as np
#import npquantor as qg
import tfquantor as qg
import os

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
tf.flags.DEFINE_string("output_dir", "quantor/", "Path to output model and checkpoint")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('frozen_pb', None, "Input frozen model to be quantized")
tf.flags.DEFINE_string('input_node_name', 'input', 'The name of the input node.')
tf.flags.DEFINE_string('output_node_name', None, 'The name of the output node.')
tf.flags.DEFINE_string('summary_dir', None, 'The directory where summaries save.')


NUM_OF_CLASSES = 151
IMAGE_SIZE = 224


def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def main(_):

  if not FLAGS.frozen_pb:
    raise ValueError('Please specify the input frozen model (with --frozen_pb flag)')
  if not FLAGS.output_node_name:
    raise ValueError('Please specify the output node name (with --output_node_name flag)')

  print("Setting up image reader...")
  train_records, _ = scene_parsing.read_dataset(FLAGS.data_dir)
  print(len(train_records))

  print("Setting up dataset reader...")
  image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
  train_dataset_reader = dataset.BatchDatset(train_records, image_options)

  print("Importing model...")
  graph_def = load_graph_def(FLAGS.frozen_pb)
  with tf.Session() as sess:
    tf.import_graph_def(graph_def, name='')
    quantized_graph = qg.create_training_graph_and_return(sess.graph)
    quantized_inf_graph = qg.create_eval_graph_and_return(sess.graph)

  with tf.Session(graph=quantized_graph) as sess:

    if FLAGS.summary_dir:
      tf.logging.info('Prepare summary writer')
      summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

    print("Setting up Saver...")
    saver = tf.train.Saver()

    logits = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES], name="logits")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                            labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                            name="entropy")))

    # initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    graph = sess.graph

    # get x and y
    x = graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))

    for var in graph.get_collection('variables'):
      varname = var.name[:-2] if var.name[-2] == ':' else var.name
      tf.summary.scalar(varname, var)
    summaries = tf.summary.merge_all()

    total_loss = 0
    for itr in xrange(FLAGS.num_batches):
      train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
      ys = sess.run(y, feed_dict={x: train_images})
      train_loss = sess.run(loss, feed_dict={logits: ys, annotation: train_annotations})
      total_loss += train_loss
      print('%d iteration: loss = %g' % (itr+1, train_loss))
      summary = sess.run(summaries)
      if FLAGS.summary_dir:
        summary_writer.add_summary(summary, itr)

    if FLAGS.summary_dir:
      summary_writer.add_graph(graph)

    # save graph and ckpts
    saver.save(sess, os.path.join(FLAGS.output_dir, "quantor.ckpt"))

  with tf.Session(graph=quantized_inf_graph) as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.output_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)

    graph_def = sess.graph.as_graph_def()
    freeze_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, [FLAGS.output_node_name])
    with open(os.path.join(FLAGS.output_dir, 'frozen.pb'), 'wb') as f:
      f.write(freeze_graph_def.SerializeToString())


if __name__ == "__main__":
  tf.app.run()
