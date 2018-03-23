.PHONY: all clean build
# mobilenet_v1
.PHONY: train_mobilenet_v1 eval_mobilenet_v1
MOBILENET_V1_EVAL_TARGETS := eval_slim_mobilenet_v1
MOBILENET_V1_EVAL_TARGETS += eval_frozen_mobilenet_v1
MOBILENET_V1_EVAL_TARGETS += freeze_mobilenet_v1
MOBILENET_V1_EVAL_TARGETS += toco_mobilenet_v1
MOBILENET_V1_EVAL_TARGETS += eval_tflite_mobilenet_v1
.PHONY: $(MOBILENET_V1_EVAL_TARGETS)
# mobilenet_v1_quantize
.PHONY: train_mobilenet_v1_quantize eval_mobilenet_v1_quantize
MOBILENET_V1_QUANTIZE_EVAL_TARGETS := eval_slim_mobilenet_v1_quantize
MOBILENET_V1_QUANTIZE_EVAL_TARGETS += freeze_mobilenet_v1_quantize
MOBILENET_V1_QUANTIZE_EVAL_TARGETS += eval_frozen_mobilenet_v1_quantize
MOBILENET_V1_QUANTIZE_EVAL_TARGETS += toco_mobilenet_v1_quantize
MOBILENET_V1_QUANTIZE_EVAL_TARGETS += eval_tflite_mobilenet_v1_quantize
.PHONY: $(MOBILENET_V1_QUANTIZE_EVAL_TARGETS)
# mobilenet_v1_quantize
.PHONY: download_google_mobilenet_v1_quantize eval_tflite_gogole_mobilenet_v1_quantize

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

ifeq ($(MOBILENET_V1_QUANT_CKPT),)
MOBILENET_V1_QUANT_CKPT := 5064184
endif

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets

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

eval_mobilenet_v1: ${MOBILENET_V1_EVAL_TARGETS}

eval_slim_mobilenet_v1:
	@ python $(TF_SLIM_BASE)/eval_image_classifier.py \
		--checkpoint_path=train_mobilenet_v1/model.ckpt-700605 \
		--dataset_dir=${DATASET_BASE}/imagenet \
		--dataset_name=imagenet \
		--dataset_split_name=validation \
		--model_name=mobilenet_v1

freeze_mobilenet_v1:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=mobilenet_v1 --dataset_name=imagenet \
		--output_file=$(ROOT_DIR)/train_mobilenet_v1/mobilenet_v1_inf_graph.pb
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=train_mobilenet_v1/mobilenet_v1_inf_graph.pb \
		--input_checkpoint=train_mobilenet_v1/model.ckpt-700605 \
		--input_binary=true \
	        --output_graph=train_mobilenet_v1/mobilenet_v1_frozen.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape

toco_mobilenet_v1:
	@ mkdir -p train_mobilenet_v1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=train_mobilenet_v1/mobilenet_v1_frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
		--output_file=train_mobilenet_v1/mobilenet_v1.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=MobilenetV1/Predictions/Reshape --input_shapes=1,224,224,3 \
		--dump_graphviz=train_mobilenet_v1/dots

eval_frozen_mobilenet_v1:
	@ echo $@
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=train_mobilenet_v1/mobilenet_v1_frozen.pb

eval_tflite_mobilenet_v1:
	@ echo $@
	@ eval_tflite \
		--summary_dir=train_mobilenet_v1/$@/summary \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=train_mobilenet_v1/mobilenet_v1.lite \
	        --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=224


# train quantize mobilenet with slim
train_mobilenet_v1_quantize:
	@ PYTHONPATH=${TF_SLIM_BASE} \
	  python train_image_classifier_quantize.py \
		--train_dir=$@ \
		--dataset_name=imagenet --dataset_split_name=train \
		--dataset_dir=${DATASET_BASE}/imagenet \
		--model_name=mobilenet_v1

eval_mobilenet_v1_quantize: ${MOBILENET_V1_QUANTIZE_EVAL_TARGETS}

eval_slim_mobilenet_v1_quantize:
	@ PYTHONPATH=${TF_SLIM_BASE} \
	  python eval_image_classifier_quantize.py \
		--checkpoint_path=train_mobilenet_v1_quantize/model.ckpt-$(MOBILENET_V1_QUANT_CKPT) \
		--dataset_dir=${DATASET_BASE}/imagenet \
		--dataset_name=imagenet \
		--dataset_split_name=validation \
		--model_name=mobilenet_v1

freeze_mobilenet_v1_quantize:
	@ PYTHONPATH=${TF_SLIM_BASE} \
	  python export_inference_graph.py \
		--alsologtostderr \
		--model_name=mobilenet_v1 --dataset_name=imagenet \
		--output_file=$(ROOT_DIR)/train_mobilenet_v1_quantize/mobilenet_v1_quantize_inf_graph.pb
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=train_mobilenet_v1_quantize/mobilenet_v1_quantize_inf_graph.pb \
		--input_checkpoint=train_mobilenet_v1_quantize/model.ckpt-$(MOBILENET_V1_QUANT_CKPT) \
		--input_binary=true \
	        --output_graph=train_mobilenet_v1_quantize/mobilenet_v1_quantize_frozen.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape

toco_mobilenet_v1_quantize:
	@ mkdir -p train_mobilenet_v1_quantize/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=train_mobilenet_v1_quantize/mobilenet_v1_quantize_frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
		--output_file=train_mobilenet_v1_quantize/mobilenet_v1_quantize.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--mean_values=128 --std_values=127 \
		--output_arrays=MobilenetV1/Predictions/Reshape --input_shapes=1,224,224,3 \
		--dump_graphviz=train_mobilenet_v1_quantize/dots

eval_frozen_mobilenet_v1_quantize:
	@ echo $@
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=train_mobilenet_v1_quantize/mobilenet_v1_quantize_frozen.pb

eval_tflite_mobilenet_v1_quantize:
	@ echo $@
	@ eval_tflite \
		--summary_dir=train_mobilenet_v1_quantize/$@/summary \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=train_mobilenet_v1_quantize/mobilenet_v1_quantize.lite \
	        --tensorflow_dir=$(TF_BASE) \
		--inference_type=uint8 \
		--max_num_batches=1000 --input_size=224

# google mobilenet v1 quantize
download_google_mobilenet_v1_quantize:
	@ mkdir -p google_mobilenet_v1_quantize
	@ cd google_mobilenet_v1_quantize \
		&& wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip
	@ cd google_mobilenet_v1_quantize \
		&& unzip mobilenet_v1_224_android_quant_2017_11_08.zip

eval_tflite_gogole_mobilenet_v1_quantize:
	@ echo $@
	@ eval_tflite \
		--summary_dir=google_mobilenet_v1_quantize/$@/summary \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=google_mobilenet_v1_quantize/mobilenet_quant_v1_224.tflite \
	        --tensorflow_dir=$(TF_BASE) \
		--inference_type=uint8 \
		--max_num_batches=1000 --input_size=224
