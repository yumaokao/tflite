from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'data_dir', None, 'The directory where the feature map files and label files are stored.')
tf.app.flags.DEFINE_integer(
    'num_batches', None, 'The number of feature map / label pairs in the directory.')

FLAGS = tf.app.flags.FLAGS

# We do not use TensorFlow code here
# due to tie issue in tf.nn.in_top_k
def evalAccuracy(pred, label):

  total = pred.shape[0]
  assert total == label.shape[0]
  top1_count = 0
  top5_count = 0

  # TODO: Use a more elegant way of numpy or TensorFlow code for this
  for i in range(total):
    cur_pred = pred[i]
    cur_label = label[i]
    idx_sort = np.argsort(cur_pred)[::-1]

    if cur_label == idx_sort[0] and cur_pred[cur_label] != cur_pred[idx_sort[1]]:
      top1_count += 1
    if cur_label in idx_sort[:5] and cur_pred[cur_label] != cur_pred[idx_sort[5]]:
      top5_count += 1

  return top1_count, top5_count, total

def main(_):
  if FLAGS.data_dir is None:
    raise ValueError('Please specify --data_dir')
  if FLAGS.num_batches is None:
    raise ValueError('Please specify --num_batches')

  top1_count = 0
  top5_count = 0
  total_count = 0

  for i in range(0, FLAGS.num_batches):

    pred = np.load(os.path.join(FLAGS.data_dir, '{}_pred.npy'.format(i)))
    label = np.load(os.path.join(FLAGS.data_dir, '{}_label.npy'.format(i)))

    top1, top5, total = evalAccuracy(pred, label)
    top1_count += top1
    top5_count += top5
    total_count += total

  print('\nResult of {} batches'.format(FLAGS.num_batches))
  print('  Top1 Accuracy: {:.4f}%'.format(100 * top1_count / float(total_count)))
  print('  Top1 Accuracy: {:.4f}%'.format(100 * top5_count / float(total_count)))

if __name__ == '__main__':
  tf.app.run()
