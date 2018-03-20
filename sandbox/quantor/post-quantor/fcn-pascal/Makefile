.PHONY: all clean build
.PHONY: download_pascal_voc_2012 download_pretrained_vgg16
.PHONY: clone_tf_image_segmentation

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
# Download and git add back, to modified for quantization

download_pascal_voc_2012:
	@ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

download_pretrained_vgg16:
	@ wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

clone_tf_image_segmentation:
	@ git clone https://github.com/warmspringwinds/tf-image-segmentation.git
