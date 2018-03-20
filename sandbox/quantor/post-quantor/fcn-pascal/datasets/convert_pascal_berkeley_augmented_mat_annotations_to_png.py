import os
import sys

sys.path.append('./tf-image-segmentation/')
from tf_image_segmentation.utils.pascal_voc import convert_pascal_berkeley_augmented_mat_annotations_to_png


pascal_berkeley_root = './datasets/benchmark_RELEASE'
convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root)
