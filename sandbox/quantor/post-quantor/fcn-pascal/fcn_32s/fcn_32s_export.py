# coding: utf-8
import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
import argparse
from tensorflow.python.framework import graph_util

parser = argparse.ArgumentParser(description='FCN 32s Export')
parser.add_argument('--ckpt_dir', default='./fcn_32s/ckpts', help='ckpt_dir')
parser.add_argument('--slim_dir', default='/home/tflite/models/research/slim', help='slim_dir')
parser.add_argument('--output_file', default='./fcn_32s/models/frozen.pb', help='output model file')
FLAGS = parser.parse_args()

sys.path.append("./tf-image-segmentation")
sys.path.append(FLAGS.slim_dir)

slim = tf.contrib.slim

from tf_image_segmentation.models.fcn_32s import FCN_32s

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
# from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive

pascal_voc_lut = pascal_segmentation_lut()

tfrecord_filename = './datasets/pascal_augmented_val.tfrecords'

number_of_classes = 21

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=1)

image, _ = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)


pred, fcn_32s_variables_mapping = FCN_32s(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=False)

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, FLAGS.ckpt_dir + "/model_fcn32s_final.ckpt")

    print("Export the Model...")
    graph_def = sess.graph.as_graph_def()
    freeze_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ["fcn_32s/prediction"])
    with open(FLAGS.output_file, 'wb') as f:
        f.write(freeze_graph_def.SerializeToString())
