.PHONY: all clean build
.PHONY: train_fcn eval_fcn toco_fcn eval_fcn_tflite

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
FCN_BASE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

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
	@ rm -rf logs

# The fcn implementaion is derived from https://github.com/shekkizh/FCN.tensorflow
# Download and git add back, to modified for quantization

train_fcn:
	@ python FCN.py

eval_fcn:
	@ mkdir -p model
	@ python eval_FCN.py \
		--output_file=$(FCN_BASE)/model/FCN_model.pb \
		--num_batches=20 \
		--batch_size=1

toco_fcn:
	@ cd $(TF_BASE) && bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(FCN_BASE)/model/FCN_model.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=$(FCN_BASE)/model/FCN_float.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=input \
		--output_arrays=inference/conv_t3 \
		--input_shapes=1,224,224,3 \
		--dump_graphviz=$(FCN_BASE)/model/dots

eval_fcn_tflite:
	@ python eval_FCN_tflite_float.py \
		--tensorflow_dir=$(TF_BASE) \
		--tflite_model=$(FCN_BASE)/model/FCN_float.lite \
		--num_batches=20 \
		--batch_size=1

quantor_fcn:
	@ mkdir -p model/quantor
	@ python quantor_FCN.py \
		--output_node_name='inference/act_quant_20/FakeQuantWithMinMaxVars' \
		--frozen_pb=$(FCN_BASE)/model/FCN_model.pb \
		--num_batches=200 \
		--batch_size=10 \
		--output_dir=$(FCN_BASE)/model/quantor \
		--summary_dir=$(FCN_BASE)/model/quantor/summary

toco_quantor_fcn:
	@ mkdir -p model/quantor/dots
	@ cd $(TF_BASE) && bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(FCN_BASE)/model/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=$(FCN_BASE)/model/quantor/FCN_uint8.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 \
		--input_arrays=input \
		--output_arrays=inference/act_quant_20/FakeQuantWithMinMaxVars \
		--input_shapes=1,224,224,3 \
		--dump_graphviz=$(FCN_BASE)/model/quantor/dots \
		--mean_values=114.8 \
		--std_values=1

eval_quantor_fcn_tflite:
	@ python eval_FCN_tflite_uint8.py \
		--tensorflow_dir=$(TF_BASE) \
		--tflite_model=$(FCN_BASE)/model/quantor/FCN_uint8.lite \
		--num_batches=20 \
		--batch_size=1 \
		--output_node_name='inference/act_quant_20/FakeQuantWithMinMaxVars'


# pretrained checkpoints could be found in swrd:/proj/mtk06790/shared/models/fcn/logs.tar.gz
