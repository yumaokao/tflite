import os
import sys

sys.path.append('./tf-image-segmentation/')
from tf_image_segmentation.utils.pascal_voc import get_augmented_pascal_image_annotation_filename_pairs
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord


pascal_root = './datasets/VOCdevkit/VOC2012'
pascal_berkeley_root = './datasets/benchmark_RELEASE'

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs = \
                get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root, 
                pascal_berkeley_root=pascal_berkeley_root,
                mode=2)

# You can create your own tfrecords file by providing
# your list with (image, annotation) filename pairs here
write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='./datasets/pascal_augmented_valeee.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='./datasets/pascal_augmented_train.tfrecords')
