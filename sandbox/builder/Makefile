.PHONY: all clean configure
.PHONY: build_tf_pip build_lite_utils dataset_imagenet
TF_BASE := /home/tflite/tensorflow
TF_MODELS_BASE := /home/tflite/models
TF_SLIM_BASE := /home/tflite/models/research/slim
DATASET_BASE := /home/tflite/datasets

all:
	@ echo "all models"

clean:
	@ cd $(TF_BASE) && bazel clean

configure:
	@ cd $(TF_BASE) && ./configure

# build tf
build_tf_pip:
	@ cd $(TF_BASE) && bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
	@ cd $(TF_BASE) && sudo pip install -U /tmp/tensorflow_pkg/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl

build_lite_utils:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite

# datasets
dataset_imagenet:
	@ cd $(TF_SLIM_BASE) && bazel build download_and_convert_imagenet
	@ DATA_DIR=$(DATASET_BASE)/imagenet \
		&& cd $(TF_SLIM_BASE) \
		&& $(TF_SLIM_BASE)/bazel-bin/download_and_convert_imagenet "$(DATASET_BASE)/imagenet"

dataset_cifar10:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=cifar10 --dataset_dir="$(DATASET_BASE)/cifar10"

dataset_flowers:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=flowers --dataset_dir="$(DATASET_BASE)/flowers"

dataset_mnist:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=mnist --dataset_dir="$(DATASET_BASE)/mnist"
