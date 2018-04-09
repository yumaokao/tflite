# coding: utf-8
import tensorflow as tf
import numpy as np
import os, sys
import argparse
import tfquantor as qg

parser = argparse.ArgumentParser(description='FCN 32s Quantor')
parser.add_argument('--frozen_pb', default=None, help='Frozen model to be quantized')
parser.add_argument('--input_node_name', default=None, help='Input node name')
parser.add_argument('--output_node_name', default=None, help='Output node name')
parser.add_argument('--slim_dir', default=None, help='slim_dir')
parser.add_argument('--output_dir', default=None, help='output_dir')
FLAGS = parser.parse_args()

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

def load_graph_def(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

graph_def = load_graph_def(FLAGS.frozen_pb)
with tf.Session() as sess:
  tf.import_graph_def(graph_def, name='')

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


  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  num_process_image = 0

  # get x and y
  x = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.input_node_name))
  y = sess.graph.get_tensor_by_name('{}:0'.format(FLAGS.output_node_name))

  for i in xrange(904):
    print('val {}'.format(i))

    image_np, shape_np, annotation_np, weight_np = sess.run([resized_and_process_input, org_shape, annotation_batch_tensor, weights])
    round_shape_np = np.round(shape_np / 32.0) * 32.0
    if np.array_equal(round_shape_np, new_shape):

      ys = sess.run(y, feed_dict={x: image_np})
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

  saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"))
