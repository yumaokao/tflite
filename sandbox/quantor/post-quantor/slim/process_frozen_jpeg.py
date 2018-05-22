from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
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
    'frozen_pb', None, 'The GraphDef file are stored with freeze_graph.')
tf.app.flags.DEFINE_string(
    'input_node_name', None, 'The name of the input node.')
tf.app.flags.DEFINE_string(
    'output_node_name', None, 'The name of the output node.')
tf.app.flags.DEFINE_integer(
    'input_size', None, 'The width/height of the input image.')
tf.app.flags.DEFINE_integer(
    'log_step', 100, 'The log step size (# batches)')

FLAGS = tf.app.flags.FLAGS
GOLDEN_COUNT = 5

def tf_float_preprocess(fns):

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
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # value range [0.0, 1.0]
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0) # resize_bilinear requires 4D shape
    image = tf.image.resize_bilinear(image, [FLAGS.input_size, FLAGS.input_size])
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5) # value range [-0.5, 0.5]
    image = tf.multiply(image, 2.0) # value range [-1.0, 1.0]

  # Make batches
  label, image = tf.train.batch([label, image], FLAGS.batch_size)
  return label, image

def tf_float_metric():
  preds_holder = tf.placeholder(tf.float32, [None, 1001 - FLAGS.labels_offset])
  labels_holder = tf.placeholder(tf.int32, [None])

  labels_holder -= FLAGS.labels_offset

  # TODO: in tf.nn.in_top_k, if multiple classes have the same prediction value
  # and straddle the top-k boundary, all of those classes are considered to be
  # in the top k, which would be weird for quantize execution (most of the value be zero)
  top1_acc, top1_acc_update = tf.metrics.accuracy(labels_holder, tf.argmax(preds_holder, 1))
  top5_acc, top5_acc_update = tf.metrics.mean(tf.nn.in_top_k(predictions=preds_holder, targets=labels_holder, k=5))
  return preds_holder, labels_holder, top1_acc, top5_acc, top1_acc_update, top5_acc_update

def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

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
  if FLAGS.frozen_pb is None:
    raise ValueError('Please specify --frozen_pb')
  if FLAGS.input_node_name is None:
    raise ValueError('Please specify --input_node_name')
  if FLAGS.output_node_name is None:
    raise ValueError('Please specify --output_node_name')
  if FLAGS.input_size is None:
    raise ValueError('Please specify --input_size')

  # Parse dataset_dir
  files = [os.path.join(FLAGS.dataset_dir, os.path.splitext(f)[0])
              for f in os.listdir(FLAGS.dataset_dir) if os.path.isfile(os.path.join(FLAGS.dataset_dir, f))]
  files = list(set(files))
  files = [os.path.abspath(f) for f in files]
  random.shuffle(files)

  # Preprocess
  next_label, next_batch = tf_float_preprocess(files)

  # Load graphdef
  graph_def = load_graph_def(FLAGS.frozen_pb)

  # Directory for all the feature map and labels
  dump_dir = os.path.dirname(FLAGS.frozen_pb)
  dump_dir = os.path.join(dump_dir, 'dump_frozen_jpeg')
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

  # Directory for the golden input/output pair
  eval_dir = os.path.join(dump_dir, 'golden')
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

  with tf.Session() as sess:

    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Import graph
    tf.import_graph_def(graph_def, name='')
    graph = sess.graph

    # Get input and output tensor
    x = graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
    y = graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))

    # Start the queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(FLAGS.num_batches):

      if i != 0 and i % FLAGS.log_step == 0:
        print('Finish processing {} batches ...'.format(i))

      label, image = sess.run([next_label, next_batch])
      pred = sess.run(y, feed_dict={x: image})

      if i < GOLDEN_COUNT:
        np.save(os.path.join(eval_dir, 'input_{}.npy'.format(i)), image)
        np.save(os.path.join(eval_dir, 'output_{}.npy'.format(i)), pred)

      np.save(os.path.join(dump_dir, '{}_pred.npy'.format(i)), pred)
      np.save(os.path.join(dump_dir, '{}_label.npy'.format(i)), label)

    print('Finish processing ALL batches')

    # Stop the queue
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  tf.app.run()
