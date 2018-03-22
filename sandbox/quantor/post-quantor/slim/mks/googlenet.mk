# float model
QUANTOR_GOOGLENET_TARGETS += eval_googlenet_frozen
# float model
QUANTOR_GOOGLENET_TARGETS += toco_googlenet
QUANTOR_GOOGLENET_TARGETS += eval_googlenet_tflite
# uint8 model
QUANTOR_GOOGLENET_TARGETS += quantor_googlenet_frozen
QUANTOR_GOOGLENET_TARGETS += toco_quantor_googlenet
QUANTOR_GOOGLENET_TARGETS += eval_quantor_googlenet_tflite

.PHONY: download_googlenet
.PHONY: ${QUANTOR_googlenet_TARGETS}
.PHONY: quantor_googlenet
.PHONY: compare_toco_googlenet_float compare_toco_googlenet_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for googlenet (batch 1) from caffe model
########################################################
download_googlenet:
	@ echo "Please manually download the model from \`\\NB17030014\Users\mtk07832\Desktop\shared\googlenet_batch1_caffe\` and rename it to \`frozen_googlenet_batch1.pb\`."

quantor_googlenet: ${QUANTOR_GOOGLENET_TARGETS}

eval_googlenet_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=prob \
		--input_node_name=data \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/googlenet/frozen_googlenet_batch1.pb --max_num_batches=10000 --batch_size=1

quantor_googlenet_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/googlenet/frozen_googlenet_batch1.pb \
		--output_node_name=prob \
		--input_node_name=data \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--output_dir=$(QUANTOR_BASE)/googlenet/quantor --max_num_batches=10000 --batch_size=1
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/googlenet/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/googlenet/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/googlenet/quantor/frozen.pb \
		--output_node_names=prob \
		--input_node_names=data
	@ save_summaries $(QUANTOR_BASE)/googlenet/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=prob \
		--input_node_name=data \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/googlenet/quantor/frozen.pb --max_num_batches=10000 --batch_size=1

toco_quantor_googlenet:
	@ mkdir -p $(QUANTOR_BASE)/googlenet/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/googlenet/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/googlenet/quantor/model.lite \
		--mean_values=114.8 --std_values=1.0 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=data \
		--default_ranges_min=0 --default_ranges_max=140 --partial_quant \
		--output_arrays=prob --input_shapes=10,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/googlenet/quantor/dots

toco_googlenet:
	@ mkdir -p $(QUANTOR_BASE)/googlenet/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/googlenet/frozen_googlenet_batch1.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/googlenet/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=data \
		--output_arrays=prob --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/googlenet/dots

eval_quantor_googlenet_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/googlenet/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/googlenet/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--labels_offset=1 --preprocess_name=vgg \
		--max_num_batches=1000 --input_size=224 --batch_size=10

eval_googlenet_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/googlenet/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/googlenet/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--labels_offset=1 --preprocess_name=vgg \
		--max_num_batches=10000 --input_size=224

