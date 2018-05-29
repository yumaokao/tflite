from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import shutil
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
  image_name, content = reader.read(fn_queue)
  image = tf.image.decode_jpeg(content, channels=3)

  if FLAGS.preprocess_name == 'inception':
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # value range [0.0, 1.0]
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0) # resize_bilinear requires 4D shape
    image = tf.image.resize_bilinear(image, [FLAGS.input_size, FLAGS.input_size])
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5) # value range [-0.5, 0.5]
    image = tf.multiply(image, 2.0) # value range [-1.0, 1.0]

  elif FLAGS.preprocess_name == 'vgg':
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    def _crop(image, offset_height, offset_width, crop_height, crop_width):
      original_shape = tf.shape(image)
      cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
      offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
      image = tf.slice(image, offsets, cropped_shape)
      return tf.reshape(image, cropped_shape)

    def _smallest_size_at_least(height, width, smallest_side):
      smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

      height = tf.to_float(height)
      width = tf.to_float(width)
      smallest_side = tf.to_float(smallest_side)

      scale = tf.cond(tf.greater(height, width),
                      lambda: smallest_side / width,
                      lambda: smallest_side / height)
      new_height = tf.to_int32(tf.rint(height * scale))
      new_width = tf.to_int32(tf.rint(width * scale))
      return new_height, new_width

    def _aspect_preserving_resize(image, smallest_side):
      smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

      shape = tf.shape(image)
      height = shape[0]
      width = shape[1]
      new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
      image = tf.expand_dims(image, 0)
      resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                               align_corners=False)
      resized_image = tf.squeeze(resized_image)
      resized_image.set_shape([None, None, 3])
      return resized_image

    def _central_crop(image, crop_height, crop_width):
      image_height = tf.shape(image)[0]
      image_width = tf.shape(image)[1]
      offset_height = (image_height - crop_height) / 2
      offset_width = (image_width - crop_width) / 2
      return _crop(image, offset_height, offset_width, crop_height, crop_width)

    def _mean_image_subtraction(image, means):
      num_channels = image.get_shape().as_list()[-1]
      channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
      for i in range(num_channels):
        channels[i] -= means[i]
      return tf.concat(axis=2, values=channels)

    image = _aspect_preserving_resize(image, 256)
    image = _central_crop(image, FLAGS.input_size, FLAGS.input_size)
    image.set_shape([FLAGS.input_size, FLAGS.input_size, 3])
    image = tf.to_float(image)
    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

  # Make batches
  label, image, image_name = tf.train.batch([label, image, image_name], FLAGS.batch_size)
  return label, image, image_name

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
  next_label, next_batch, next_origin = tf_float_preprocess(files)

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

      label, image, image_name = sess.run([next_label, next_batch, next_origin])
      pred = sess.run(y, feed_dict={x: image})

      if i < GOLDEN_COUNT:
        # store the first image in the batch
        org_0 = image_name[0:1]
        img_0 = image[0:1]
        pred_0 = pred[0:1]
        shutil.copy(org_0[0], os.path.join(eval_dir, 'origin_{}.jpeg'.format(i)))
        np.save(os.path.join(eval_dir, 'input_{}.npy'.format(i)), img_0)
        np.save(os.path.join(eval_dir, 'output_{}.npy'.format(i)), pred_0)

      np.save(os.path.join(dump_dir, '{}_pred.npy'.format(i)), pred)
      np.save(os.path.join(dump_dir, '{}_label.npy'.format(i)), label)

    print('Finish processing ALL batches')

    # Stop the queue
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  tf.app.run()
