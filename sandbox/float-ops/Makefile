.PHONY: all build clean

TF_BASE := /home/tflite/tensorflow
OPS_BASE := /home/tflite/sandbox/float-ops

VARIABLE_TARGETS := fully_connected concat multiply
VARIABLE_TARGETS += conv2d depthwise_conv2d conv2d_transpose
FIXED_TARGETS := avg_pool max_pool l2_pool
FIXED_TARGETS += relu relu6 sigmoid tanh softmax floor
FIXED_TARGETS += resize_bilinear
FIXED_TARGETS += space_to_depth depth_to_space
FIXED_TARGETS += l2_normalize local_response_normalization

# float-ops not passed with compare_tflite
#    1. no tflite kernels:
# 	floor
#
#    2. --allow_custom_ops:
#       conv2d_transpose -> TransposeConv


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

define compare_tflite
	@ echo "compare_tflite"
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/compare_tflite \
		--tflite_file=$1/export/model.lite \
		--batch_xs=$1/export/batch_xs.npy \
		--batch_ys=$1/export/ys.npy
endef


all:
	@ echo "all float-ops"
	@ echo $(VARIABLE_TARGETS) | tr '\n' '\0' | xargs -d ' ' -i make {}
	@ echo $(FIXED_TARGETS) | tr '\n' '\0' | xargs -d ' ' -i make {}

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite


$(VARIABLE_TARGETS):
	@ mkdir -p $@/export
	@ python float-ops.py $@
	$(call freeze_graph,$(OPS_BASE)/$@,ys)
	$(call toco,$(OPS_BASE)/$@,frozen.pb,Placeholder,ys,)
	$(call compare_tflite,$(OPS_BASE)/$@)

$(FIXED_TARGETS):
	@ mkdir -p $@/export
	@ python float-ops.py $@
	$(call toco,$(OPS_BASE)/$@,model.pb,Placeholder,ys)
	$(call compare_tflite,$(OPS_BASE)/$@)
