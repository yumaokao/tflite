.PHONY: all clean build
.PHONY: dataset_imagenet dataset_cifar10 dataset_flowers dataset_mnist
.PHONY: train_cifarnet eval_cifarnet freeze_graph

TF_BASE := /home/tflite/tensorflow
TF_SLIM_BASE := /home/tflite/models/research/slim
DATASET_BASE := /home/tflite/datasets
QUANTOR_BASE := /home/tflite/sandbox/quantor
TOOLS_BASE := /home/tflite/sandbox/mnist/tools


all:
	@ echo "all models"

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:summarize_graph

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


# slim cifarnet
#   add --clone_on_cpu to run training on cpu
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

eval_cifarnet:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/cifarnet --eval_dir=$(QUANTOR_BASE)/cifarnet \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 --model_name=cifarnet

freeze_cifarnet:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr --model_name=cifarnet \
		--output_file=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/cifarnet/model.ckpt-100000 \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb \
		--output_node_names=CifarNet/Predictions/Reshape
