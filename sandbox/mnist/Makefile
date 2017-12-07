.PHONY: all fc fc_sin deep deep_dummy

TF_BASE := /home/tflite/tensorflow
MNIST_BASE := /home/tflite/sandbox/mnist

## functions ##
define mnist_freeze_graph
	@ echo "freeze_graph"
	@ # echo MNIST_DIR [$1]
	@ # echo OUTPUT_NODE_NAMES $2
	$(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$1/mnist.pb \
		--input_checkpoint=$1/model.ckpt \
		--input_binary=true \
		--output_graph=$1/frozen_mnist.pb \
		--output_node_names=$2
endef

define mnist_toco
	@ echo "toco"
	@ # echo MNIST_DIR [$1]
	@ # echo MNIST_PB [$2]
	@ # echo OUTPUT_ARRAYS $3
	@ # echo $4
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/mnist.lite \
		--inference_type=FLOAT \
		--influrence_input_type=FLOAT --input_arrays=Placeholder \
		--output_arrays=$3 --input_shapes=10,784 \
		--dump_graphviz=$1/dots $4
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef

define mnist_toco_dummy_quant
	@ echo "toco"
	@ # echo MNIST_DIR [$1]
	@ # echo MNIST_PB [$2]
	@ # echo OUTPUT_ARRAYS $3
	@ # echo $4
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/mnist.lite \
		--inference_type=QUANTIZED_UINT8 \
		--influrence_input_type=QUANTIZED_UINT8 --input_arrays=Placeholder \
		--output_arrays=$3 --input_shapes=10,784 \
		--default_ranges_min=0 --default_ranges_max=6 \
		--mean_values=127.5 --std_values=127.5 \
		--dump_graphviz=$1/dots $4
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef


all:
	@ echo "Do nothing"

fc:
	@ mkdir -p $@/export
	@ python fc/mnist_softmax.py
	@ echo "freeze_graph"
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,add)
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist.pb,add,)

fc_sin:
	@ mkdir -p $@/export
	@ python fc_sin/mnist_softmax.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,add_1)
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist.pb,add_1,--allow_custom_ops)

deep:
	@ mkdir -p $@/export
	@ python $@/mnist_deep.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,fc2/add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist-nodropout.pb,fc2/add,)

deep_dummy: deep
	@ mkdir -p $@/export
	@ python $@/mnist_deep.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,fc2/add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	$(call mnist_toco_dummy_quant,$(MNIST_BASE)/$@,frozen_mnist-nodropout.pb,fc2/add,)
