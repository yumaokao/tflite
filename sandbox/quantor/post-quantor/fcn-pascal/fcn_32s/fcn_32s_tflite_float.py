# coding: utf-8
import tensorflow as tf
import numpy as np
import os, sys
import argparse
import subprocess

parser = argparse.ArgumentParser(description='FCN 32s Float TFLite')
parser.add_argument('--tensorflow_dir', default=None, help='Path to tensorflow')
parser.add_argument('--tflite_model', default=None, help='Input float tflite model')
parser.add_argument('--slim_dir', default='/home/tflite/models/research/slim', help='slim_dir')
FLAGS = parser.parse_args()

if not FLAGS.tensorflow_dir:
    raise ValueError('You must supply the tensorflow directory with --tensorflow_dir')
if not FLAGS.tflite_model:
    raise ValueError('You must supply the frozen pb with --tflite_model')

sys.path.append("./tf-image-segmentation")
sys.path.append(FLAGS.slim_dir)

slim = tf.contrib.slim

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
# from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive

pascal_voc_lut = pascal_segmentation_lut()

tfrecord_filename = './datasets/pascal_augmented_val.tfrecords'

number_of_classes = 21
vgg_mean = [123.680, 116.779, 103.939]
new_shape = [384, 512]

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=1)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image, axis=0)
annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

# Take away the masked out values from evaluation
weights = tf.to_float( tf.not_equal(annotation_batch_tensor, 255) )

# replace annotation 255 with 0, since already has weight mask
annotation_batch_tensor = tf.where(tf.equal(annotation_batch_tensor, 255),
                                   tf.zeros_like(annotation_batch_tensor),
                                   annotation_batch_tensor)

# Resize the input
org_shape = tf.shape(image_batch_tensor)
org_shape = org_shape[1:3]
resized_input = tf.image.resize_images(image_batch_tensor, new_shape)
resized_and_process_input = resized_input - vgg_mean

# Resize the output
logits_holder = tf.placeholder(tf.float32, [None, None, None, None])
pred = tf.argmax(logits_holder, axis=3)
pred = tf.expand_dims(pred, axis=3)
pred = tf.image.resize_nearest_neighbor(images=pred, size=org_shape)

# Define the accuracy metric: Mean Intersection Over Union
miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                   labels=annotation_batch_tensor,
                                                   num_classes=number_of_classes,
                                                   weights=weights)

eval_dir = os.path.dirname(FLAGS.tflite_model)
run_tflite_cmd = [FLAGS.tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(FLAGS.tflite_model),
          '--batch_xs={}'.format(os.path.join(eval_dir, 'batch_xs.npy')),
          '--batch_ys={}'.format(os.path.join(eval_dir, 'output_ys.npy')),
          '--inference_type=float']

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

with tf.Session() as sess:

    sess.run(initializer)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    num_process_image = 0

    # There are 904 images in restricted validation dataset
    for i in xrange(904):
        print('val {}'.format(i))

        image_np, shape_np, annotation_np, weight_np = sess.run([resized_and_process_input, org_shape, annotation_batch_tensor, weights])

        round_shape_np = np.round(shape_np / 32.0) * 32.0
        if np.array_equal(round_shape_np, new_shape):
            np.save(os.path.join(eval_dir, 'batch_xs.npy'), image_np)
            subprocess.check_output(run_tflite_cmd)
            ys = np.load(os.path.join(eval_dir, 'output_ys.npy'))
            _ = sess.run(update_op, feed_dict={logits_holder: ys, annotation_batch_tensor: annotation_np, weights: weight_np, org_shape: shape_np})
            num_process_image += 1

        # Display the image and the segmentation result
        # upsampled_predictions = pred_np.squeeze()
        #plt.imshow(image_np)
        #plt.show()
        #visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut)

    coord.request_stop()
    coord.join(threads)

    res = sess.run(miou)

    print("Pascal VOC 2012 Restricted (RV-VOC12) Mean IU: " + str(res) + '(' + str(num_process_image) + ' images)')

