.PHONY: all clean build
.PHONY: dataset_imagenet dataset_cifar10 dataset_flowers dataset_mnist

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
QUANTOR_BASE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

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
	@ rm -rf $(QUANTOR_BASE)/cifarnet


# datasets
dataset_imagenet:
	@ cd $(TF_SLIM_BASE) && bazel build download_and_convert_imagenet
	@ DATA_DIR=$(DATASET_BASE)/imagenet \
		&& cd $(TF_SLIM_BASE) \
		&& $(TF_SLIM_BASE)/bazel-bin/download_and_convert_imagenet "$(DATASET_BASE)/imagenet"

dataset_cifar10:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=cifar10 --dataset_dir="$(DATASET_BASE)/cifar10"

dataset_flowers:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=flowers --dataset_dir="$(DATASET_BASE)/flowers"

dataset_mnist:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=mnist --dataset_dir="$(DATASET_BASE)/mnist"

########################################################
# slim cifarnet
########################################################
include mks/cifarnet.mk

########################################################
# for mobilenetV1_224
########################################################
include mks/mobilenetv1.mk

########################################################
# for inception_v3
########################################################
include mks/inception_v3.mk

########################################################
# for inception_v1
########################################################
include mks/inception_v1.mk

########################################################
# for inception_resnet_v2
########################################################
include mks/inception_resnet_v2.mk

########################################################
# for resnet_v1
########################################################
include mks/resnet_v1_50.mk

########################################################
# for resnet_v2
########################################################
include mks/resnet_v2_50.mk

########################################################
# for vgg_16
########################################################
include mks/vgg_16.mk

########################################################
# for googlenet (batch 1) from caffe model
########################################################
include mks/googlenet.mk

########################################################
# for mobilenet (+ softmax layer) from caffe model
########################################################
include mks/mobilenet_softmax.mk

########################################################
# for resnet34
########################################################
include mks/resnet34.mk
