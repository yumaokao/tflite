from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


def prepare_cifar10_dataset(filenames, width, height,
                            batch_size=1, inference_type='float'):
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

  def _preprocessing_cifarnet(image, label):
    tf.summary.image('image', tf.expand_dims(image, 0))
    image = tf.to_float(image)
    image = tf.image.resize_image_with_crop_or_pad(image, width, height)
    tf.summary.image('resized_image', tf.expand_dims(image, 0))
    image = tf.image.per_image_standardization(image)
    tf.summary.image('std_image', tf.expand_dims(image, 0))
    return image, label

  # YMK: use _preprocessing_imagenet [-1, 1) is easier
  #      for toco with --mean_value=127.5 --std_value=127.5
  def _preprocessing_imagenet(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

  # tf.Dataset
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)
  if inference_type == 'float':
    dataset = dataset.map(_preprocessing_imagenet)
  dataset = dataset.batch(batch_size)
  return dataset


def prepare_imagenet_dataset(filenames, width, height,
                             batch_size=1, inference_type='float'):
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

  def _preprocessing(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [width, height],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

  def _resize_imagenet(image, label):
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_nearest_neighbor(image, [width, height],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image, label

  # tf.Dataset
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)
  if inference_type == 'float':
    dataset = dataset.map(_preprocessing)
  else:
    dataset = dataset.map(_resize_imagenet)
  dataset = dataset.batch(batch_size)
  return dataset


def prepare_tfrecords(dataset_name, dataset_dir, dataset_split_name):
  with tf.name_scope("tfrecords"):
    if dataset_name == 'imagenet':
      # TODO: more portable name
      return [os.path.join(dataset_dir, 'validation-{:05d}-of-00128'.format(i))
              for i in range(0, 128)]
    elif dataset_name == 'cifar10':
      return [os.path.join(dataset_dir, '{}_{}.tfrecord'.format(
                           dataset_name, dataset_split_name))]
    else:
      tf.logging.error('Could not found preprocessing for dataset {}'.format(dataset_name))
      return None


def prepare_dataset(filenames, dataset_name, input_size,
                    batch_size=1, inference_type='float'):
  with tf.name_scope("datasets"):
    if dataset_name == 'imagenet':
      return prepare_imagenet_dataset(filenames, input_size, input_size,
                                      batch_size=batch_size,
                                      inference_type=inference_type)
    elif dataset_name == 'cifar10':
      return prepare_cifar10_dataset(filenames, 32, 32,
                                     batch_size=batch_size,
                                     inference_type=inference_type)
    else:
      tf.logging.error('Could not found preprocessing for dataset {}'.format(dataset_name))
      return None


def prepare_metrics(dataset_name, inference_type='float'):
  with tf.name_scope("metrics"):
    if dataset_name == 'imagenet':
      pred_shape = [None, 1001]
    elif dataset_name == 'cifar10':
      pred_shape = [None, 10]
    else:
      tf.logging.error('Could not found metrics for dataset {}'.format(dataset_name))
      return None
    lbls = tf.placeholder(tf.int32, [None, 1])
    if inference_type == 'float':
      preds = tf.placeholder(tf.int32, pred_shape)
    elif inference_type == 'uint8':
      preds = tf.placeholder(tf.uint8, pred_shape)
    accuracy, acc_update_op = tf.metrics.accuracy(lbls, tf.argmax(preds, 1))
    tf.summary.scalar('accuracy', accuracy)
    return lbls, preds, accuracy, acc_update_op
