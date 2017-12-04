.PHONY: all fc fc_sin

TF_BASE := /home/tflite/tensorflow
MNIST_BASE := /home/tflite/sandbox/mnist


all:
	@ echo "Do nothing"

fc:
	@ python fc/mnist_softmax.py
	@ echo "freeze_graph"
	@ $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(MNIST_BASE)/fc/mnist.pb \
		--input_checkpoint=$(MNIST_BASE)/fc/model.ckpt \
		--input_binary=true \
		--output_graph=$(MNIST_BASE)/fc/frozen_mnist.pb \
		--output_node_names=add
	@ echo "toco"
	@ mkdir -p $(MNIST_BASE)/fc/export
	@ mkdir -p $(MNIST_BASE)/fc/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(MNIST_BASE)/fc/frozen_mnist.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(MNIST_BASE)/fc/export/mnist.lite \
		--inference_type=FLOAT \
		--influrence_input_type=FLOAT --input_arrays=Placeholder \
		--output_arrays=add --input_shapes=10,784 \
		--dump_graphviz=$(MNIST_BASE)/fc/dots
	# @ cd $(MNIST_BASE)/fc_sin/dots && dot -Tpdf -O ./toco_*.dot

fc_sin:
	@ python fc_sin/mnist_softmax.py
	@ echo "freeze_graph"
	@ $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(MNIST_BASE)/fc_sin/mnist.pb \
		--input_checkpoint=$(MNIST_BASE)/fc_sin/model.ckpt \
		--input_binary=true \
		--output_graph=$(MNIST_BASE)/fc_sin/frozen_mnist.pb \
		--output_node_names=add_1
	@ echo "toco"
	@ mkdir -p $(MNIST_BASE)/fc_sin/export
	@ mkdir -p $(MNIST_BASE)/fc_sin/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(MNIST_BASE)/fc_sin/frozen_mnist.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(MNIST_BASE)/fc_sin/export/mnist.lite \
		--inference_type=FLOAT \
		--influrence_input_type=FLOAT --input_arrays=Placeholder \
		--output_arrays=add_1 --input_shapes=10,784 \
		--dump_graphviz=$(MNIST_BASE)/fc_sin/dots \
		--allow_custom_ops
	# @ cd $(MNIST_BASE)/fc_sin/dots && dot -Tpdf -O ./toco_*.dot
