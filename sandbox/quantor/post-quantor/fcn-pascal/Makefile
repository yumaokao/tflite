.PHONY: all clean build
.PHONY: download_dataset prepare_dataset convert_tfrecords
.PHONY: download_pretrained_vgg16
.PHONY: clone_tf_image_segmentation pip_install_requirements

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow

all:
	@ echo "all models"

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:summarize_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:transform_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite

clean:
	@ rm -rf logs

# The fcn implementaion is derived from https://github.com/warmspringwinds/tf-image-segmentation
# But clone https://github.com/barryridge/tf-image-segmentation.git for newer tensorflow support
# Download and git add back, to modified for quantization

# in docker with tensorflow
pip_install_requirements:
	@ pip install --user scikit-image

download_dataset:
	@ mkdir -p datasets
	@ cd datasets && wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
	@ cd datasets && wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

prepare_dataset:
	@ cd datasets && tar xvf VOCtrainval_11-May-2012.tar
	@ cd datasets && tar zxvf benchmark.tgz

# in docker with tensorflow
convert_tfrecords:
	@ python datasets/datasets/convert_pascal_berkeley_augmented_mat_annotations_to_png.py
	@ python datasets/convert_pascal_voc_tfrecords.py

download_pretrained_vgg16:
	@ mkdir -p vgg_16_ckpts
	@ cd vgg_16_ckpts && wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
	@ cd vgg_16_ckpts && tar zxvf vgg_16_2016_08_28.tar.gz

clone_tf_image_segmentation:
	# @ git clone https://github.com/warmspringwinds/tf-image-segmentation.git
	@ git clone https://github.com/barryridge/tf-image-segmentation.git
