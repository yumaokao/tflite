.PHONY: dataset_imagenet dataset_cifar10 dataset_flowers dataset_mnist
.PHONY: train_cifarnet
.PHONY: eval_cifarnet

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

all:
	@ echo "all models"

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

# slim train
train_cifarnet:
	@ cd $(TF_SLIM_BASE) && python train_image_classifier.py \
		--train_dir=$(QUANTOR_BASE)/cifarnet --dataset_name=cifar10 \
		--dataset_split_name=train --dataset_dir=$(DATASET_BASE)/cifar10 \
		--model_name=cifarnet --preprocessing_name=cifarnet \
		--max_number_of_steps=100000 --batch_size=128 \
		--save_interval_secs=120 --save_summaries_secs=120 \
		--log_every_n_steps=100 --optimizer=sgd \
		--learning_rate=0.1 --learning_rate_decay_factor=0.1 \
		--num_epochs_per_decay=200 --weight_decay=0.004

# slim eval
eval_cifarnet:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/cifarnet --eval_dir=./cifarnet-model \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 --model_name=cifarnet
