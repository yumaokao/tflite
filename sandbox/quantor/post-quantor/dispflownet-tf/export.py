import sys
import os
import time
import logging
import argparse
import numpy as np
from dispnet import DispNet
import tensorflow as tf

from tensorflow.python.framework import graph_util

INPUT_SIZE = (384, 768, 3)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--ckpt", dest="checkpoint_path", required=True, type=str,
          metavar="FILE", help='model checkpoint path')
  parser.add_argument("-o", "--output", dest="output_path", required=True, type=str,
            metavar="FILE", help='path to output frozen model')
  parser.add_argument('--use_corr', action='store_true', default=False)

  args = parser.parse_args()

  dispnet = DispNet(mode="inference", ckpt_path=args.checkpoint_path,
                    input_size=INPUT_SIZE, is_corr=args.use_corr)

  ckpt = tf.train.latest_checkpoint(args.checkpoint_path)

  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(graph=dispnet.graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(dispnet.init)
    dispnet.saver.restore(sess=sess, save_path=ckpt)
    print("Restoring from %s" % ckpt)

    freeze_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['prediction/conv/BiasAdd'])
    if not os.path.exists(args.output_path):
      os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, 'frozen.pb'), 'wb') as f:
      f.write(freeze_graph_def.SerializeToString())

