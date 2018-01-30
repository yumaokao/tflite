.PHONY: train_cifarnet eval_cifarnet freeze_graph eval_cifarnet_frozen
.PHONY: toco_quantor_cifarnet toco_cifarnet eval_cifarnet_tflite eval_quantor_cifarnet_tflite

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# slim cifarnet
#   add --clone_on_cpu to run training on cpu
########################################################
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
		--alsologtostderr \
		--model_name=cifarnet --dataset_name=cifar10 \
		--output_file=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ save_summaries $(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/cifarnet/model.ckpt-100000 \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb \
		--output_node_names=CifarNet/Predictions/Reshape
	@ save_summaries $(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb

# eval frozen
eval_cifarnet_frozen:
	@ eval_frozen \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--output_node_name=CifarNet/Predictions/Reshape \
		--frozen_pb=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb

quantor_cifarnet_frozen:
	@ quantor_frozen \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--output_node_name=CifarNet/Predictions/Reshape \
		--frozen_pb=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb \
		--output_dir=$(QUANTOR_BASE)/cifarnet/quantor
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/cifarnet/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/cifarnet/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/cifarnet/quantor/frozen.pb \
		--output_node_names=CifarNet/Predictions/Reshape
	@ save_summaries $(QUANTOR_BASE)/cifarnet/quantor/frozen.pb
	@ eval_frozen \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/quantor/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--output_node_name=CifarNet/Predictions/Reshape \
		--frozen_pb=$(QUANTOR_BASE)/cifarnet/quantor/frozen.pb

# TODO(yumaokao): LRN to be appened a FakeQuant
#   CifarNet/conv1/Relu
#   CifarNet/Predictions/Reshape
toco_quantor_cifarnet:
	@ mkdir -p $(QUANTOR_BASE)/cifarnet/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/cifarnet/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/cifarnet/quantor/model.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--mean_values=128 --std_values=127 \
		--output_arrays=CifarNet/Predictions/Reshape --input_shapes=1,32,32,3 \
		--default_ranges_min=0 --default_ranges_max=10 --partial_quant \
		--dump_graphviz=$(QUANTOR_BASE)/cifarnet/quantor/dots

toco_cifarnet:
	@ mkdir -p $(QUANTOR_BASE)/cifarnet/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/cifarnet/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=CifarNet/Predictions/Reshape --input_shapes=1,32,32,3 \
		--dump_graphviz=$(QUANTOR_BASE)/cifarnet/dots

eval_quantor_cifarnet_tflite:
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/quantor/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--tflite_model=$(QUANTOR_BASE)/cifarnet/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE)

eval_cifarnet_tflite:
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/quantor/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--tflite_model=$(QUANTOR_BASE)/cifarnet/float_model.lite --tensorflow_dir=$(TF_BASE)
