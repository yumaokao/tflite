.PHONY: all clean
.PHONY: fc fc_sin fc_fake
.PHONY: deep deep_dummy deep_custom deep_lrn deep_lrn_dummy
.PHONY: lstm

TF_BASE := /home/tflite/tensorflow
MNIST_BASE := /home/tflite/sandbox/mnist

## functions ##
define mnist_freeze_graph
	@ echo "freeze_graph"
	@ # echo MNIST_DIR [$1]
	@ # echo OUTPUT_NODE_NAMES $2
	@ $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$1/mnist.pb \
		--input_checkpoint=$1/ckpts/model.ckpt \
		--input_binary=true \
		--output_graph=$1/frozen_mnist.pb \
		--output_node_names=$2
endef

define mnist_toco
	@ echo "toco"
	@ # echo MNIST_DIR [$1]
	@ # echo MNIST_PB [$2]
	@ # echo INPUT_ARRAYS $3
	@ # echo OUTPUT_ARRAYS $4
	@ # echo $5
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/mnist.lite \
		--inference_type=FLOAT \
		--influrence_input_type=FLOAT --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=10,784 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef

define mnist_toco_dummy_quant
	@ echo "toco"
	@ # echo MNIST_DIR [$1]
	@ # echo MNIST_PB [$2]
	@ # echo INPUT_ARRAYS $3
	@ # echo OUTPUT_ARRAYS $4
	@ # echo $5
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/mnist.lite \
		--inference_type=QUANTIZED_UINT8 \
		--influrence_input_type=QUANTIZED_UINT8 --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=10,784 \
		--default_ranges_min=0 --default_ranges_max=6 \
		--mean_values=127.5 --std_values=127.5 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef

define mnist_toco_quant
	@ echo "toco"
	@ # echo MNIST_DIR [$1]
	@ # echo MNIST_PB [$2]
	@ # echo INPUT_ARRAYS $3
	@ # echo OUTPUT_ARRAYS $4
	@ # echo $5
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/mnist.lite \
		--inference_type=QUANTIZED_UINT8 \
		--influrence_input_type=QUANTIZED_UINT8 --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=10,784 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef


all:
	@ echo "Do nothing"

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb

# simple a fully connected network
fc:
	@ mkdir -p $@/export
	@ python $@/mnist_softmax.py
	@ echo "freeze_graph"
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,add)
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist.pb,Placeholder,add,)

# a fully connected network with fakequant
fc_fake:
	@ mkdir -p $@/export
	@ python $@/mnist_softmax.py
	@ echo "freeze_graph"
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,y_q)
	$(call mnist_toco_quant,$(MNIST_BASE)/$@,frozen_mnist.pb,x_q,y_q,)

# simple a fully connected network and a sin op
fc_sin:
	@ mkdir -p $@/export
	@ python $@/mnist_softmax.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,add_1)
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist.pb,Placeholder,add_1,--allow_custom_ops)

# lenet
deep:
	@ mkdir -p $@/export
	@ python $@/mnist_deep.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,fc2/add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist-nodropout.pb,Placeholder,fc2/add,)

# lenet with dummy quant
deep_dummy:
	@ mkdir -p $@/export
	@ python $@/mnist_deep.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,fc2/add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	$(call mnist_toco_dummy_quant,$(MNIST_BASE)/$@,frozen_mnist-nodropout.pb,Placeholder,fc2/add,)

# lenet with Conv2D -> CustomConv2D
deep_custom:
	@ mkdir -p $@/export
	@ python $@/mnist_deep.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,fc2/add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	@ python tools/rename_ops.py $@/frozen_mnist-nodropout.pb -i 15,CustomConv2D
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist-nodropout-custom.pb,Placeholder,fc2/add,--allow_custom_ops)

# lenet with a lrn between conv1 and conv2
deep_lrn:
	@ mkdir -p $@/export
	@ python $@/mnist_deep_lrn.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,fc2/add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist-nodropout.pb,Placeholder,fc2/add,)

# lenet with a lrn between conv1 and conv2, and dummy quant (will failed)
deep_lrn_dummy:
	@ mkdir -p $@/export
	@ python $@/mnist_deep_lrn.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,fc2/add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	$(call mnist_toco_dummy_quant,$(MNIST_BASE)/$@,frozen_mnist-nodropout.pb,Placeholder,fc2/add,--partial_quant --allow_custom_ops)

# lenet with a lrn between conv1 and conv2, and dummy quant (will failed)
lstm:
	@ mkdir -p $@/export
	@ python $@/recurrent_network.py
	$(call mnist_freeze_graph,$(MNIST_BASE)/$@,add)
	@ python tools/drop_dropouts.py $@/frozen_mnist.pb
	@ python tools/save_summaries.py $@/frozen_mnist-nodropout.pb
	$(call mnist_toco,$(MNIST_BASE)/$@,frozen_mnist-nodropout.pb,Placeholder,add,--allow_custom_ops)
