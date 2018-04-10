#!/usr/bin/env python
# coding=utf-8
"""
This is a script for downloading and converting the pascal voc 2012 dataset
and the berkely extended version.

    # original PASCAL VOC 2012
    # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB

    # berkeley augmented Pascal VOC
    # wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB

This can be run as an independent executable to download
the dataset or be imported by scripts used for larger experiments.

If you aren't sure run this to do a full download + conversion setup of the dataset:
   ./data_pascal_voc.py pascal_voc_setup
"""
from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
import sys
import numpy as np
from PIL import Image
from collections import defaultdict
import os
from keras.utils import get_file
# from tf_image_segmentation.recipes import datasets
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord
from tf_image_segmentation.utils import pascal_voc
import tarfile

# ============== Ingredient 2: dataset =======================
data_pascal_voc = Experiment("dataset")


@data_pascal_voc.config
def voc_config():
    # TODO(ahundt) add md5 sums for each file
    verbose = True
    dataset_root = os.path.expanduser("~") + "/.keras/datasets"
    dataset_path = dataset_root + '/VOC2012'
    # sys.path.append("tf-image-segmentation/")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # based on https://github.com/martinkersner/train-DeepLab

    # original PASCAL VOC 2012
    # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
    pascal_root = dataset_path + '/VOCdevkit/VOC2012'

    # berkeley augmented Pascal VOC
    # wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB

    # Pascal Context
    # http://www.cs.stanford.edu/~roozbeh/pascal-context/
    # http://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz
    pascal_berkeley_root = dataset_path + '/benchmark_RELEASE'
    urls = [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/33_context_labels.tar.gz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/59_context_labels.tar.gz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/33_labels.txt',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt'
    ]
    filenames = ['VOCtrainval_11-May-2012.tar',
                 'benchmark.tgz',
                 'trainval.tar.gz',
                 '33_context_labels.tar.gz',
                 '59_context_labels.tar.gz',
                 '33_labels.txt',
                 '59_labels.txt'
                 ]

    md5s = ['6cd6e144f989b92b3379bac3b3de84fd',
            '82b4d87ceb2ed10f6038a1cba92111cb',
            'df034edb2c12aa7d33b42b20bb1796e3',
            '180101cfc01c71867b6686207f071eb9',
            'f85d450010762a0e1080304286ce30ed',
            '8840f5439b471aecf991ac6448b826e6',
            '993901f2d930cc038c406845f08fa082']

    combined_imageset_train_txt = dataset_path + '/combined_imageset_train.txt'
    combined_imageset_val_txt = dataset_path + '/combined_imageset_val.txt'
    combined_annotations_path = dataset_path + '/combined_annotations'

    # see get_augmented_pascal_image_annotation_filename_pairs()
    voc_data_subset_mode = 2


@data_pascal_voc.capture
def pascal_voc_files(dataset_path, filenames, dataset_root, urls, md5s):
    print(dataset_path)
    print(dataset_root)
    print(urls)
    print(filenames)
    print(md5s)
    return [dataset_path + filename for filename in filenames]


@data_pascal_voc.command
def pascal_voc_download(dataset_path, filenames, dataset_root, urls, md5s):
    zip_paths = pascal_voc_files(dataset_path, filenames, dataset_root, urls, md5s)
    for url, filename, md5 in zip(urls, filenames, md5s):
        path = get_file(filename, url, md5_hash=md5, extract=True, cache_subdir=dataset_path)



@data_pascal_voc.command
def convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root):
    pascal_voc.convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root)


@data_pascal_voc.config
def cfg_pascal_voc_segmentation_to_tfrecord(dataset_path, filenames, dataset_root):
    tfrecords_train_filename = dataset_path + '/pascal_augmented_train.tfrecords'
    tfrecords_val_filename = dataset_path + '/pascal_augmented_val.tfrecords'


@data_pascal_voc.command
def pascal_voc_berkeley_combined(dataset_path,
                                 pascal_root,
                                 pascal_berkeley_root,
                                 voc_data_subset_mode,
                                 combined_imageset_train_txt,
                                 combined_imageset_val_txt,
                                 combined_annotations_path):
    # Returns a list of (image, annotation)
    # filename pairs (filename.jpg, filename.png)
    overall_train_image_annotation_filename_pairs, \
        overall_val_image_annotation_filename_pairs = \
        pascal_voc.get_augmented_pascal_image_annotation_filename_pairs(
            pascal_root=pascal_root,
            pascal_berkeley_root=pascal_berkeley_root,
            mode=voc_data_subset_mode)
    # combine the annotation files into one folder
    pascal_voc.pascal_combine_annotation_files(
        overall_train_image_annotation_filename_pairs +
        overall_val_image_annotation_filename_pairs,
        combined_annotations_path)
    # generate the train imageset txt
    pascal_voc.pascal_filename_pairs_to_imageset_txt(
        combined_imageset_train_txt,
        overall_train_image_annotation_filename_pairs
    )
    # generate the val imageset txt
    pascal_voc.pascal_filename_pairs_to_imageset_txt(
        combined_imageset_val_txt,
        overall_val_image_annotation_filename_pairs
    )


@data_pascal_voc.command
def pascal_voc_segmentation_to_tfrecord(dataset_path,
                                        pascal_root,
                                        pascal_berkeley_root,
                                        voc_data_subset_mode,
                                        tfrecords_train_filename,
                                        tfrecords_val_filename):
    # Returns a list of (image, annotation)
    # filename pairs (filename.jpg, filename.png)
    overall_train_image_annotation_filename_pairs, \
        overall_val_image_annotation_filename_pairs = \
        pascal_voc.get_augmented_pascal_image_annotation_filename_pairs(
            pascal_root=pascal_root,
            pascal_berkeley_root=pascal_berkeley_root,
            mode=voc_data_subset_mode)

    # You can create your own tfrecords file by providing
    # your list with (image, annotation) filename pairs here
    #
    # this will create a tfrecord in:
    # tf_image_segmentation/tf_image_segmentation/recipes/pascal_voc/
    write_image_annotation_pairs_to_tfrecord(
        filename_pairs=overall_val_image_annotation_filename_pairs,
        tfrecords_filename=tfrecords_val_filename)

    write_image_annotation_pairs_to_tfrecord(
        filename_pairs=overall_train_image_annotation_filename_pairs,
        tfrecords_filename=tfrecords_train_filename)


@data_pascal_voc.command
def pascal_voc_setup(filenames, dataset_path, pascal_root,
                     pascal_berkeley_root, dataset_root,
                     voc_data_subset_mode,
                     tfrecords_train_filename,
                     tfrecords_val_filename,
                     urls, md5s,
                     combined_imageset_train_txt,
                     combined_imageset_val_txt,
                     combined_annotations_path):
    # download the dataset
    pascal_voc_download(dataset_path, filenames,
                        dataset_root, urls, md5s)
    # convert the relevant files to a more useful format
    convert_pascal_berkeley_augmented_mat_annotations_to_png(
        pascal_berkeley_root)
    pascal_voc_berkeley_combined(dataset_path,
                                 pascal_root,
                                 pascal_berkeley_root,
                                 voc_data_subset_mode,
                                 combined_imageset_train_txt,
                                 combined_imageset_val_txt,
                                 combined_annotations_path)
    pascal_voc_segmentation_to_tfrecord(dataset_path, pascal_root,
                                        pascal_berkeley_root,
                                        voc_data_subset_mode,
                                        tfrecords_train_filename,
                                        tfrecords_val_filename)


@data_pascal_voc.automain
def main(filenames, dataset_path, pascal_root,
         pascal_berkeley_root, dataset_root,
         voc_data_subset_mode,
         tfrecords_train_filename,
         tfrecords_val_filename,
         urls, md5s,
         combined_imageset_train_txt,
         combined_imageset_val_txt,
         combined_annotations_path):
    voc_config()
    pascal_voc_setup(filenames, dataset_path, pascal_root,
                     pascal_berkeley_root, dataset_root,
                     voc_data_subset_mode,
                     tfrecords_train_filename,
                     tfrecords_val_filename,
                     urls, md5s,
                     combined_imageset_train_txt,
                     combined_imageset_val_txt,
                     combined_annotations_path)