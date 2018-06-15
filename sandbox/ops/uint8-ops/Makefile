.PHONY: all build clean

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
OPS_BASE := $(TFLITE_ROOT_PATH)/sandbox/ops/uint8-ops

UINT8_TARGETS := depth_to_space
VARIABLE_TARGETS := conv2d depthwise_conv2d fully_connected
VARIABLE_TARGETS += atrous_conv2d conv2d_transpose
VARIABLE_TARGETS += concat add multiply
FIXED_TARGETS := softmax max_pool avg_pool space_to_depth
FIXED_TARGETS += resize_bilinear mean leaky_relu slice pad

.PHONY: $(UINT8_TARGETS)
.PHONY: $(VARIABLE_TARGETS)
.PHONY: $(FIXED_TARGETS)

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

define freeze_graph
	@ echo "freeze_graph"
	$(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$1/model.pb \
		--input_checkpoint=$1/ckpts/model.ckpt \
		--input_binary=true \
		--output_graph=$1/frozen.pb \
		--output_node_names=$2
endef

define toco_uint8
	@ echo "toco"
	@ rm -rf $1/dots
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/model.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=10,784 \
		--dump_graphviz=$1/dots $5 \
		--mean_values=63.75 \
		--std_values=63.75 \
		--allow_custom_ops
	@ cd $1/dots && dot -Tpdf -O ./toco_*.dot
endef

define compare_tflite
	@ echo "compare_tflite"
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/run_tflite \
		--inference_type="uint8" \
		--tflite_file=$1/export/model.lite \
		--batch_xs=$1/export/batch_xs.npy \
		--batch_ys=$1/export/output_ys.npy
	@ compare_npys -v \
		--references $1/export/ys.npy \
		--computeds $1/export/output_ys.npy
endef


all:
	@ echo "all uint8-ops"
	# @ echo $(UINT8_TARGETS) | tr '\n' '\0' | xargs -d ' ' -i make {}
	@ echo $(VARIABLE_TARGETS) | tr '\n' '\0' | xargs -d ' ' -i make {}
	@ echo $(FIXED_TARGETS) | tr '\n' '\0' | xargs -d ' ' -i make {}

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb */__pycache__

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite

$(VARIABLE_TARGETS):
	@ mkdir -p $@/export
	@ python uint8-ops.py $@ # $@/$@.py
	$(call freeze_graph,$(OPS_BASE)/$@,ys)
	$(call toco_uint8,$(OPS_BASE)/$@,frozen.pb,Placeholder,ys)
	$(call compare_tflite,$(OPS_BASE)/$@)

$(FIXED_TARGETS):
	@ mkdir -p $@/export
	@ python uint8-ops.py $@ # $@/$@.py
	$(call toco_uint8,$(OPS_BASE)/$@,model.pb,Placeholder,ys)
	$(call compare_tflite,$(OPS_BASE)/$@)

