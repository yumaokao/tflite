# float model
QUANTOR_MOBILENET_SOFTMAX_TARGETS := freeze_mobilenet_softmax
QUANTOR_MOBILENET_SOFTMAX_TARGETS += eval_mobilenet_softmax_frozen
# float model
QUANTOR_MOBILENET_SOFTMAX_TARGETS += toco_mobilenet_softmax
QUANTOR_MOBILENET_SOFTMAX_TARGETS += eval_mobilenet_softmax_tflite
# uint8 model
QUANTOR_MOBILENET_SOFTMAX_TARGETS += quantor_mobilenet_softmax_frozen
QUANTOR_MOBILENET_SOFTMAX_TARGETS += toco_quantor_mobilenet_softmax
QUANTOR_MOBILENET_SOFTMAX_TARGETS += eval_quantor_mobilenet_softmax_tflite

.PHONY: download_mobilenet_softmax eval_mobilenet_softmax
.PHONY: ${QUANTOR_MOBILENET_SOFTMAX_TARGETS}
.PHONY: freeze_mobilenet_softmax eval_mobilenet_softmax_frozen
.PHONY: quantor_mobilenet_softmax_frozen toco_mobilenet_softmax
.PHONY: eval_mobilenet_softmax_tflite
.PHONY: compare_toco_mobilenet_softmax_float compare_toco_mobilenet_softmax_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for mobilenet (+ softmax layer) from caffe model
########################################################
download_mobilenet_softmax:
	@ echo "Please manually download the model from \`\\NB17030014\Users\mtk07832\Desktop\shared\mobilenet_softmax_caffe\`."

eval_mobilenet_softmax_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=prob \
		--input_size=224 --batch_size=1 \
		--labels_offset=1 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_softmax/frozen_mobilenet_softmax.pb --max_num_batches=10000 \
		# --summary_dir=$(QUANTOR_BASE)/mobilenet_softmax/summary/$@

quantor_mobilenet_softmax_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_softmax/frozen_mobilenet_softmax.pb \
		--output_node_name=prob --batch_size=1 \
		--input_size=224 --labels_offset=1 \
		--output_dir=$(QUANTOR_BASE)/mobilenet_softmax/quantor --max_num_batches=10000
		# --summary_dir=$(QUANTOR_BASE)/mobilenet_softmax/summary/$@
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/mobilenet_softmax/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/mobilenet_softmax/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/mobilenet_softmax/quantor/frozen.pb \
		--output_node_names=prob
	@ save_summaries $(QUANTOR_BASE)/mobilenet_softmax/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=prob --batch_size=1 \
		--input_size=224 --labels_offset=1 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_softmax/quantor/frozen.pb --max_num_batches=10000
		# --summary_dir=$(QUANTOR_BASE)/mobilenet_softmax/quantor/summary/$@

toco_quantor_mobilenet_softmax:
	@ echo "Should resolve issues when add fakequant ops"
	# @ mkdir -p $(QUANTOR_BASE)/mobilenet_softmax/dots
	# @ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
	#	--input_file=$(QUANTOR_BASE)/mobilenet_softmax/quantor/frozen.pb \
	#	--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
	#	--output_file=$(QUANTOR_BASE)/mobilenet_softmax/quantor/model.lite \
	#	--mean_values=128 --std_values=127 \
	#	--inference_type=QUANTIZED_UINT8 \
	#	--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
	#	--output_arrays=prob --input_shapes=1,224,224,3 \
	#	--dump_graphviz=$(QUANTOR_BASE)/mobilenet_softmax/dots

eval_quantor_mobilenet_softmax_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/mobilenet_softmax/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/mobilenet_softmax/quantor/model.lite --tensorflow_dir=$(TF_BASE) \
		--inference_type=uint8 \
		--max_num_batches=10000 --input_size=224 --batch_size=1

toco_mobilenet_softmax:
	@ mkdir -p $(QUANTOR_BASE)/mobilenet_softmax/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/mobilenet_softmax/frozen_mobilenet_softmax.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/mobilenet_softmax/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=prob --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/mobilenet_softmax/dots

eval_mobilenet_softmax_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/mobilenet_softmax/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/mobilenet_softmax/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=10000 --input_size=224 --labels_offset=1
