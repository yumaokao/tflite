from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


def prepare_cifar10_dataset(filenames, width, height,
                            preprocess_name='inception', batch_size=1,
                            labels_offset=0, inference_type='float'):
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

  # YMK: use _preprocessing_inception [-1, 1) is easier
  #      for toco with --mean_value=127.5 --std_value=127.5
  def _preprocessing_inception(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

  # YMK: use _preprocessing_vgg for vgg/resnet
  #      for toco with --mean_value=114.8 --std_value=1.0
  #      slim eval choose _R_MEAN = 123.68, _G_MEAN = 116.78, _B_MEAN = 103.94
  #      however per_channel not yet supported, so _ALL_MEAN = 114.8
  def _preprocessing_vgg(image, label):
    image = tf.to_float(image)
    image = tf.expand_dims(image, 0)
    image = tf.subtract(image, 114.8)
    return image, label

  def _preprocessing_vgg_official(image, label):
    image = tf.to_float(image)
    image = tf.expand_dims(image, 0)
    image = tf.subtract(image, 114.8)
    image = image / 255.0
    return image, label

  # proprocessing func
  _preprocessing_func = None
  if inference_type == 'float':
    if preprocess_name == 'inception':
      _preprocessing_func = _preprocessing_inception
    elif preprocess_name == 'vgg':
      _preprocessing_func = _preprocessing_vgg
    elif preprocess_name == 'vgg_official':
      _preprocessing_func = _preprocessing_vgg_official

  # tf.Dataset
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)
  if _preprocessing_func is not None:
    dataset = dataset.map(_preprocessing_func)
  dataset = dataset.batch(batch_size)
  return dataset


def prepare_imagenet_dataset(filenames, width, height,
                             preprocess_name='inception', batch_size=1,
                             labels_offset=0, inference_type='float'):
  def _read_tfrecord(example_proto):
    feature_to_type = {
        "image/class/label": tf.FixedLenFeature([1], dtype=tf.int64),
        "image/encoded": tf.FixedLenFeature([], dtype=tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    label = parsed_features["image/class/label"]
    rawpng = parsed_features["image/encoded"]
    image_decoded = tf.image.decode_png(rawpng, channels=3)
    label -= labels_offset
    return image_decoded, label

  def _preprocessing_inception(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [width, height],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

  # YMK: use _preprocessing_vgg for vgg/resnet
  #      for toco with --mean_value=-114.8 --std_value=1.0
  #      slim eval choose _R_MEAN = 123.68, _G_MEAN = 116.78, _B_MEAN = 103.94
  #      however per_channel not yet supported, so _ALL_MEAN = 114.8
  def _preprocessing_vgg(image, label):
    image = tf.to_float(image)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [width, height],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 114.8)
    return image, label

  def _preprocessing_vgg_official(image, label):
    image = tf.to_float(image)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [width, height],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 114.8)
    image = image / 255.0
    return image, label

  def _resize_imagenet(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [width, height],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image, label

  # proprocessing func
  _preprocessing_func = None
  if inference_type == 'float':
    if preprocess_name == 'inception':
      _preprocessing_func = _preprocessing_inception
    elif preprocess_name == 'vgg':
      _preprocessing_func = _preprocessing_vgg
    elif preprocess_name == 'vgg_official':
      _preprocessing_func = _preprocessing_vgg_official
  elif inference_type == 'uint8':
    _preprocessing_func = _resize_imagenet

  # tf.Dataset
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_read_tfrecord)
  dataset = dataset.map(_preprocessing_func)
  dataset = dataset.batch(batch_size)
  return dataset


def prepare_tfrecords(dataset_name, dataset_dir, dataset_split_name):
  with tf.name_scope("tfrecords"):
    if dataset_name not in ['imagenet', 'cifar10']:
      tf.logging.error('Could not find tfrecords for dataset {}'.format(dataset_name))
      return None
    if dataset_split_name not in ['train', 'test']:
      tf.logging.error('Could not find tfrecords for dataset_split_name {}'.format(dataset_split_name))
      return None

    if dataset_name == 'imagenet':
      if dataset_split_name == 'test':
        return [os.path.join(dataset_dir, 'validation-{:05d}-of-00128'.format(i))
                for i in range(0, 128)]
      else:
        return [os.path.join(dataset_dir, 'train-{:05d}-of-01024'.format(i))
                for i in range(0, 1024)]
    elif dataset_name == 'cifar10':
      return [os.path.join(dataset_dir, '{}_{}.tfrecord'.format(
                           dataset_name, dataset_split_name))]


def prepare_dataset(filenames, dataset_name, input_size,
                    preprocess_name='inception', batch_size=1,
                    labels_offset=0, inference_type='float'):
  with tf.name_scope("datasets"):
    if dataset_name not in ['imagenet', 'cifar10']:
      tf.logging.error('Could not find preprocessing for dataset {}'.format(dataset_name))
      return None
    if preprocess_name not in ['inception', 'vgg', 'vgg_official']:
      tf.logging.error('Could not find preprocessing method {}'.format(preprocess_name))
      return None
    if inference_type not in ['float', 'uint8']:
      tf.logging.error('Could not find preprocessing method {}'.format(preprocess_name))
      return None

    if dataset_name == 'imagenet':
      return prepare_imagenet_dataset(filenames, input_size, input_size,
                                      preprocess_name=preprocess_name,
                                      batch_size=batch_size,
                                      labels_offset=labels_offset,
                                      inference_type=inference_type)
    elif dataset_name == 'cifar10':
      return prepare_cifar10_dataset(filenames, 32, 32,
                                     preprocess_name=preprocess_name,
                                     batch_size=batch_size,
                                     labels_offset=labels_offset,
                                     inference_type=inference_type)


def prepare_metrics(dataset_name, labels_offset=0, inference_type='float'):
  with tf.name_scope("metrics"):
    if dataset_name not in ['imagenet', 'cifar10']:
      tf.logging.error('Could not find metrics for dataset {}'.format(dataset_name))
      return None
    if inference_type not in ['float', 'uint8']:
      tf.logging.error('Could not find preprocessing method {}'.format(preprocess_name))
      return None

    if dataset_name == 'imagenet':
      pred_shape = [None, 1001 - labels_offset]
    elif dataset_name == 'cifar10':
      pred_shape = [None, 10 - labels_offset]
    lbls = tf.placeholder(tf.int32, [None, 1])
    if inference_type == 'float':
      preds = tf.placeholder(tf.float32, pred_shape)
    elif inference_type == 'uint8':
      preds = tf.placeholder(tf.uint8, pred_shape)
    accuracy, acc_update_op = tf.metrics.accuracy(lbls, tf.argmax(preds, 1))
    tf.summary.scalar('accuracy', accuracy)
    return lbls, preds, accuracy, acc_update_op
