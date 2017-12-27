.PHONY: all build clean

TF_BASE := /home/tflite/tensorflow
MODELS_BASE := /home/tflite/sandbox/models
TOOLS_BASE := /home/tflite/sandbox/mnist/tools

VARIABLE_TARGETS := mobilenet_1.0

.PHONY: $(VARIABLE_TARGETS)

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
		--output_arrays=$4 --input_shapes=10,224,224,3 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef

define compare_tflite
	@ echo "compare_tflite"
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/compare_tflite \
		--tflite_file=$1/export/model.lite \
		--batch_xs=$1/export/batch_xs.npy \
		--batch_ys=$1/export/ys.npy $2
endef


all:
	@ echo "all models"

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite


mobilenet_1.0:
	@ mkdir -p $@/export
	@ cp backbone/mobilenet_1.0/mobilenet_v1_1.0_224_frozen.pb $@/frozen.pb
	@ python $(TOOLS_BASE)/drop_dropouts.py $@/frozen.pb
	@ python $@/$@.py $@/frozen-nodropout.pb
	$(call toco,$(MODELS_BASE)/$@,frozen-nodropout.pb,input,MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd,)
	$(call compare_tflite,$(MODELS_BASE)/$@,--ignore 1)
