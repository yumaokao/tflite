from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files (i.e., tfrecord) are stored.')
tf.app.flags.DEFINE_string(
    'data_prefix', None, 'Prefix string of the tfrecord files.')
tf.app.flags.DEFINE_string(
    'output_dir', None, 'The output directory of the generated image files.')
tf.app.flags.DEFINE_string(
    'label_file', None, 'File contains the mapping between imagenet class ID to human readable labels.')
tf.app.flags.DEFINE_integer(
    'count', None, 'Number of images to generate. Set -1 for converting all the images')
FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = 1000

def _read_tfrecord(example_proto):
  feature_to_type = {
    "image/class/label": tf.FixedLenFeature([1], dtype=tf.int64),
    "image/encoded": tf.FixedLenFeature([], dtype=tf.string)
  }
  parsed_features = tf.parse_single_example(example_proto, feature_to_type)
  label = parsed_features["image/class/label"]
  rawpng = parsed_features["image/encoded"]
  image_decoded = tf.image.decode_png(rawpng, channels=3)
  return image_decoded, label

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('Please specify --dataset_dir')
  if not FLAGS.label_file:
    raise ValueError('Please specify --label_file')
  if not FLAGS.output_dir:
    raise ValueError('Please specify --output_dir')
  if FLAGS.count is None:
    raise ValueError('Please specify --count')
  if not os.path.isfile(FLAGS.label_file):
    raise ValueError('Specified --label_file is NOT a file')
  if not os.path.isdir(FLAGS.dataset_dir):
    raise ValueError('Specified --dataset_dir does NOT exist')
  if not os.path.isdir(FLAGS.output_dir):
    raise ValueError('Specified --output_dir does NOT exist')

  label_map = {}
  with open(FLAGS.label_file, 'r') as f:
    for line in f.readlines():
      line = line.rstrip()
      idx, label = line.split(':')
      assert idx not in label_map
      label_map[int(idx)] = label

  files = [f for f in os.listdir(FLAGS.dataset_dir) if os.path.isfile(os.path.join(FLAGS.dataset_dir, f))]
  if FLAGS.data_prefix is not None:
    files = [f for f in files if f.startswith(FLAGS.data_prefix)]

  files = [os.path.join(FLAGS.dataset_dir, f) for f in files]

  dataset = tf.data.TFRecordDataset(files)
  dataset = dataset.map(_read_tfrecord)
  iterator = dataset.make_one_shot_iterator()
  next_batch = iterator.get_next()

  image_holder = tf.placeholder(tf.uint8, shape=[None, None, 3])
  to_jpg = tf.image.encode_jpeg(image_holder, format='rgb')

  # count of all the classes appeared in the converted images
  dist = [0] * (NUM_CLASSES + 1)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    try:
      if FLAGS.count == -1:
        iterator = itertools.count(0)
      else:
        iterator = range(FLAGS.count)

      for i in iterator:
        npy_image, label = sess.run(next_batch)
        label = label.flatten()
        assert label.size == 1
        cur_label = label[0]
        dist[cur_label] += 1

        print('{} -> label: {},  image shape: {}, image type: {}'.format(i, cur_label, npy_image.shape, npy_image.dtype))
        jpg_image = sess.run(to_jpg, feed_dict={image_holder: npy_image})

        out_fn = str(dist[cur_label])

        if not os.path.exists(os.path.join(FLAGS.output_dir, str(cur_label))):
          os.mkdir(os.path.join(FLAGS.output_dir, str(cur_label)))

        with open(os.path.join(FLAGS.output_dir, str(cur_label), out_fn + '.jpeg'), 'w') as f:
          f.write(jpg_image)
        with open(os.path.join(FLAGS.output_dir, str(cur_label), out_fn + '.log'), 'w') as f:
          f.write('{}\n{}\n'.format(label[0], label_map[label[0]]))

    except tf.errors.OutOfRangeError:
      if FLAGS.count == -1:
        print('Total {} images is converted'.format(i))
      else:
        print('Request {} images, but only {} images are found'.format(FLAGS.count, i))

if __name__ == "__main__":
    tf.app.run()
