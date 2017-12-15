.PHONY: all clean

TF_BASE := /home/tflite/tensorflow
OPS_BASE := /home/tflite/sandbox/float-ops

VARIABLE_TARGETS := fully_connected
FIXED_TARGETS := avg_pool max_pool
FIXED_TARGETS += relu softmax sigmoid tanh l2_normalize relu6

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
		--influrence_input_type=FLOAT --input_arrays=$3 \
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
	@ echo "Do nothing"

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb

# simple a fully connected network
$(VARIABLE_TARGETS):
	@ mkdir -p $@/export
	@ python $@/$@.py
	$(call freeze_graph,$(OPS_BASE)/$@,add)
	$(call toco,$(OPS_BASE)/$@,frozen.pb,Placeholder,add,)
	$(call compare_tflite,$(OPS_BASE)/$@)

$(FIXED_TARGETS):
	@ mkdir -p $@/export
	@ python float-ops.py $@
	$(call toco,$(OPS_BASE)/$@,model.pb,Placeholder,ys)
	$(call compare_tflite,$(OPS_BASE)/$@)
