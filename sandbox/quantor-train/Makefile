.PHONY: all clean build
.PHONY: train_mobilenet_v1 eval_mobilenet_v1
.PHONY: train_mobilenet_v1_quantize eval_mobilenet_v1_quantize

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
TOOLS_BASE := $(TFLITE_ROOT_PATH)/sandbox/mnist/tools

all:
	@ echo "all models"

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:summarize_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite

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

# train mobilenet with slim
train_mobilenet_v1:
	@ python $(TF_SLIM_BASE)/train_image_classifier.py \
		--train_dir=$@ \
		--dataset_name=imagenet --dataset_split_name=train \
		--dataset_dir=${DATASET_BASE}/imagenet \
		--model_name=mobilenet_v1

eval_mobilenet_v1:
	@ python $(TF_SLIM_BASE)/eval_image_classifier.py \
		--checkpoint_path=train_mobilenet_v1/model.ckpt-700605 \
		--dataset_dir=${DATASET_BASE}/imagenet \
		--dataset_name=imagenet \
		--dataset_split_name=validation \
		--model_name=mobilenet_v1

# train quantize mobilenet with slim
train_mobilenet_v1_quantize:
	@ PYTHONPATH=${TF_SLIM_BASE} \
	  python train_image_classifier_quantize.py \
		--train_dir=$@ \
		--dataset_name=imagenet --dataset_split_name=train \
		--dataset_dir=${DATASET_BASE}/imagenet \
		--model_name=mobilenet_v1

eval_mobilenet_v1_quantize:
	@ PYTHONPATH=${TF_SLIM_BASE} \
	  python eval_image_classifier_quantize.py \
		--checkpoint_path=train_mobilenet_v1_quantize/model.ckpt-156822 \
		--dataset_dir=${DATASET_BASE}/imagenet \
		--dataset_name=imagenet \
		--dataset_split_name=validation \
		--model_name=mobilenet_v1
