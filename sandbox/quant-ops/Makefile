.PHONY: all build clean

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
OPS_BASE := $(TFLITE_ROOT_PATH)/sandbox/quant-ops

QUANT_TARGETS := dequantize quantize_v2 quantize
.PHONY: $(QUANT_TARGETS)

## functions ##
define freeze_graph
	@ echo "freeze_graph"
	$(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$1/model.pb \
		--input_checkpoint=$1/ckpts/model.ckpt \
		--input_binary=true \
		--output_graph=$1/frozen.pb \
		--output_node_names=$2
endef

define toco
	@ echo "toco"
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=10,784 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef

define toco_dequantize
	@ echo "toco"
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/model.lite \
		--inference_type=FLOAT \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=10,784 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef

define compare_tflite
	@ echo "compare_tflite"
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/compare_tflite \
		--tflite_file=$1/export/model.lite \
		--batch_xs=$1/export/batch_xs.npy \
		--batch_ys=$1/export/ys.npy
endef


all:
	@ echo "all quant-ops"
	@ echo $(QUANT_TARGETS) | tr '\n' '\0' | xargs -d ' ' -i make {}

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite

dequantize:
	@ mkdir -p $@/export
	@ python $@/$@.py
	$(call toco_dequantize,$(OPS_BASE)/$@,model.pb,Placeholder,ys)
	$(call compare_tflite,$(OPS_BASE)/$@)

quantize_v2 quantize:
	@ mkdir -p $@/export
	@ python $@/$@.py
	$(call toco,$(OPS_BASE)/$@,model.pb,Placeholder,ys)
	$(call compare_tflite,$(OPS_BASE)/$@)
