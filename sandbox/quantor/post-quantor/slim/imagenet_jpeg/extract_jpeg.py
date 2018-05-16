from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os
import itertools
import random
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'jpeg_dir', None, 'The directory where the jpeg image files are stored.')
tf.app.flags.DEFINE_string(
    'output_dir', None, 'The output directory of the generated image files.')
tf.app.flags.DEFINE_string(
    'output_prefix', None, 'Prefix string of the generated image files.')
tf.app.flags.DEFINE_integer(
    'count', None, 'Number of images to generate.')
FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 1000

def main(_):
  if not FLAGS.jpeg_dir:
    raise ValueError('Please specify --jpeg_dir')
  if not FLAGS.output_dir:
    raise ValueError('Please specify --output_dir')
  if not FLAGS.output_prefix:
    raise ValueError('Please specify --output_prefix')
  if FLAGS.count is None:
    raise ValueError('Please specify --count')
  if not os.path.isdir(FLAGS.jpeg_dir):
    raise ValueError('Specified --jpeg_dir does NOT exist')
  if not os.path.isdir(FLAGS.output_dir):
    raise ValueError('Specified --output_dir does NOT exist')

  total = 0
  filenames = []
  for i in range(1, NUM_CLASSES + 1):
    cur_dir = os.path.join(FLAGS.jpeg_dir, str(i))
    if os.path.exists(cur_dir):
      files = [os.path.splitext(f)[0] for f in os.listdir(cur_dir) if os.path.isfile(os.path.join(cur_dir, f))]
      files = [os.path.join(cur_dir, f) for f in files]
      files = list(set(files))
      random.shuffle(files)
    else:
      files = []

    filenames.append((i, files))
    total += len(files)

  random.shuffle(filenames)
  if FLAGS.count > total:
    raise ValueError('Only {} images are found in the directory (request {} images)'.format(total, FLAGS.count))

  dist = [0] * (NUM_CLASSES + 1)
  source_names = []
  idx = 0
  cur_count = 0
  while True:
    if cur_count == FLAGS.count:
      break
    if len(filenames[idx % NUM_CLASSES][1]) > 0:
      dist[filenames[idx % NUM_CLASSES][0]] += 1
      source_names.append(filenames[idx % NUM_CLASSES][1][0])
      del filenames[idx % NUM_CLASSES][1][0]
      cur_count += 1
    idx += 1

  for i in range(1, NUM_CLASSES + 1):
    print('label #{}: {} images'.format(i, dist[i]))

  num_digits = len(str(FLAGS.count))
  for i in range(1, FLAGS.count+1):
    if FLAGS.output_prefix:
      dstname = FLAGS.output_prefix + '_{0:0{1}d}'.format(i, num_digits)
    else:
      dstname = '{0:0{1}d}'.format(i, num_digits)
    dstname = os.path.join(FLAGS.output_dir, dstname)

    shutil.copyfile(source_names[i-1] + '.jpeg', dstname + '.jpeg')
    shutil.copyfile(source_names[i-1] + '.log', dstname + '.log')


if __name__ == "__main__":
    tf.app.run()
