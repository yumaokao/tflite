# float model
QUANTOR_MOBILENETV1_224_TARGETS := freeze_mobilenet_v1_224
QUANTOR_MOBILENETV1_224_TARGETS += eval_mobilenet_v1_224_frozen
# float model
QUANTOR_MOBILENETV1_224_TARGETS += toco_mobilenet_v1_224
QUANTOR_MOBILENETV1_224_TARGETS += eval_mobilenet_v1_224_tflite
# uint8 model
QUANTOR_MOBILENETV1_224_TARGETS += quantor_mobilenet_v1_224_frozen
QUANTOR_MOBILENETV1_224_TARGETS += toco_quantor_mobilenet_v1_224
QUANTOR_MOBILENETV1_224_TARGETS += eval_quantor_mobilenet_v1_224_tflite

.PHONY: download_mobilenet_v1_224 eval_mobilenet_v1_224
.PHONY: ${QUANTOR_MOBILENETV1_224_TARGETS}
.PHONY: freeze_mobilenet_v1_224 eval_mobilenet_v1_224_frozen
.PHONY: quantor_mobilenet_v1_224_frozen toco_mobilenet_v1_224
.PHONY: eval_mobilenet_v1_224_tflite
.PHONY: compare_toco_mobilenet_v1_224_float compare_toco_mobilenet_v1_224_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for mobilenet_v1_224
########################################################
download_mobilenet_v1_224:
	@ wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz -P $(QUANTOR_BASE)/mobilenet_v1_224
	@ tar xvf $(QUANTOR_BASE)/mobilenet_v1_224/mobilenet_v1_1.0_224_2017_06_14.tar.gz -C $(QUANTOR_BASE)/mobilenet_v1_224

eval_mobilenet_v1_224:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/mobilenet_v1_224/mobilenet_v1_1.0_224.ckpt \
		--eval_dir=$(QUANTOR_BASE)/mobilenet_v1_224 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=mobilenet_v1 --max_num_batches=200

quantor_mobilenet_v1_224: $(QUANTOR_MOBILENETV1_224_TARGETS)

# sub targets
freeze_mobilenet_v1_224:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=mobilenet_v1 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/mobilenet_v1_224/mobilenet_v1_224_inf_graph.pb
	@ save_summaries $(QUANTOR_BASE)/mobilenet_v1_224/mobilenet_v1_224_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/mobilenet_v1_224/mobilenet_v1_224_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/mobilenet_v1_224/mobilenet_v1_1.0_224.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/mobilenet_v1_224/frozen_mobilenet_v1_224.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/mobilenet_v1_224/frozen_mobilenet_v1_224.pb
	@ save_summaries $(QUANTOR_BASE)/mobilenet_v1_224/frozen_mobilenet_v1_224.pb

eval_mobilenet_v1_224_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_v1_224/frozen_mobilenet_v1_224.pb --max_num_batches=200 \
		# --summary_dir=$(QUANTOR_BASE)/mobilenet_v1_224/summary/$@

quantor_mobilenet_v1_224_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_v1_224/frozen_mobilenet_v1_224.pb \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--output_dir=$(QUANTOR_BASE)/mobilenet_v1_224/quantor --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/mobilenet_v1_224/summary/$@
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/frozen.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape
	@ save_summaries $(QUANTOR_BASE)/mobilenet_v1_224/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/frozen.pb --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/summary/$@

toco_quantor_mobilenet_v1_224:
	@ mkdir -p $(QUANTOR_BASE)/mobilenet_v1_224/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/model.lite \
		--mean_values=128 --std_values=127 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=MobilenetV1/Predictions/Reshape --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/mobilenet_v1_224/dots

toco_mobilenet_v1_224:
	@ mkdir -p $(QUANTOR_BASE)/mobilenet_v1_224/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/mobilenet_v1_224/frozen_mobilenet_v1_224.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/mobilenet_v1_224/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=MobilenetV1/Predictions/Reshape --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/mobilenet_v1_224/dots

eval_quantor_mobilenet_v1_224_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/model.lite --tensorflow_dir=$(TF_BASE) \
		--inference_type=uint8 \
		--max_num_batches=1000 --input_size=224 --batch_size=10

eval_mobilenet_v1_224_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/mobilenet_v1_224/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=10000 --input_size=224

########################################################
# compare_toco
########################################################
compare_toco_mobilenet_v1_224_float:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_v1_224/frozen_mobilenet_v1_224.pb \
		--max_num_batches=100 \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=224 \
		--dump_data=False

compare_toco_mobilenet_v1_224_uint8:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenet_v1_224/quantor/frozen.pb \
		--max_num_batches=100 \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=uint8 \
		--input_size=224 \
		--dump_data=False
