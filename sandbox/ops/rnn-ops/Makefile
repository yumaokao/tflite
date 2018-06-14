.PHONY: all build clean

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
OPS_BASE := $(TFLITE_ROOT_PATH)/sandbox/ops/rnn-ops
TOOLS_BASE := $(TFLITE_ROOT_PATH)/sandbox/mnist/tools

RNN_TARGETS := basic_lstm_cell gru_cell basic_rnn_cell
.PHONY: $(RNN_TARGETS)

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
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/run_tflite \
		--inference_type="float" \
		--tflite_file=$1/export/model.lite \
		--batch_xs=$1/export/batch_xs.npy \
		--batch_ys=$1/export/output_ys.npy
	@ compare_npys -v \
		--references $1/export/ys.npy \
		--computeds $1/export/output_ys.npy
endef


all:
	@ echo "all rnn-ops"
	@ echo $(RNN_TARGETS) | tr '\n' '\0' | xargs -d ' ' -i make {}

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite

basic_lstm_cell:
	@ mkdir -p $@/export
	@ python $@/$@.py
	$(call freeze_graph,$(OPS_BASE)/$@,add)
	@ mkdir -p $(OPS_BASE)/$@/dots
	$(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(OPS_BASE)/$@/frozen_mnist-nodropout.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(OPS_BASE)/$@/export/mnist.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=Placeholder,BasicLSTMCellZeroState/zeros_1	\
		--output_arrays=add --input_shapes=10,784:64,128 \
		--dump_graphviz=$(OPS_BASE)/$@/dots --allow_custom_ops \
		--rnn_states="mnist"
#	$(call compare_tflite,$(OPS_BASE)/$@)

basic_rnn_cell:
	@ mkdir -p $@/export
	@ python $@/$@.py
	$(call freeze_graph,$(OPS_BASE)/$@,add)
	@ python $(TOOLS_BASE)/drop_dropouts.py $@/frozen.pb
	@ python $(TOOLS_BASE)/save_summaries.py $@/frozen-nodropout.pb
	@ mkdir -p $(OPS_BASE)/$@/dots
	$(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(OPS_BASE)/$@/frozen-nodropout.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(OPS_BASE)/$@/export/mnist.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=Placeholder,BasicLSTMCellZeroState/zeros_1	\
		--output_arrays=add --input_shapes=10,784:64,128 \
		--dump_graphviz=$(OPS_BASE)/$@/dots --allow_custom_ops

gru_cell:
	@ mkdir -p $@/export
	@ python $@/$@.py
	$(call freeze_graph,$(OPS_BASE)/$@,add)
	@ python $(TOOLS_BASE)/drop_dropouts.py $@/frozen.pb
	@ python $(TOOLS_BASE)/save_summaries.py $@/frozen-nodropout.pb
	@ mkdir -p $(OPS_BASE)/$@/dots
	$(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(OPS_BASE)/$@/frozen-nodropout.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(OPS_BASE)/$@/export/mnist.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=Placeholder,GRUCellZeroState/zeros \
		--output_arrays=add --input_shapes=10,784:64,128 \
		--dump_graphviz=$(OPS_BASE)/$@/dots --allow_custom_ops

# speakerid to trace and compare
speakerid:
	@ mkdir -p $@/export
	@ [ -f $@/export/speakerid.tflite ] || \
		wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_speakerid_model_2017_11_14.tflite \
		-O $@/export/speakerid.tflite
	@ mkdir -p $(OPS_BASE)/$@/dots
	$(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(OPS_BASE)/$@/export/speakerid.tflite \
		--input_format=TFLITE --output_format=GRAPHVIZ_DOT \
		--output_file=$(OPS_BASE)/$@/export/mnist.dot \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=Placeholder \
		--output_arrays=add --input_shapes=10,784 \
		--dump_graphviz=$(OPS_BASE)/$@/dots --allow_custom_ops
