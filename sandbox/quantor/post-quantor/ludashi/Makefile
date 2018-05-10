ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

DATASET_BASE:=$(TFLITE_ROOT_PATH)/datasets
TF_BASE:=$(TFLITE_ROOT_PATH)/tensorflow

all:
	@ echo "all models"

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite


include mks/vgg16.mk
include mks/inception_v3.mk
include mks/resnet34.mk
