.PHONY: all clean build
.PHONY: pip_install_dep protobuf_compilation test_installation download_pets

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
TF_MODELS_BASE := $(TFLITE_ROOT_PATH)/models
TF_RESEARCH_BASE := $(TF_MODELS_BASE)/research
TF_OBJDECT_BASE := $(TF_RESEARCH_BASE)/object_detection
TF_SLIM_BASE := $(TF_RESEARCH_BASE)/slim

all:
	@ echo "all models"

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:summarize_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:transform_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite

clean:
	@ rm -rf *.tar.gz


# object detection api install.md
# run this in docker env.
pip_install_dep:
	@ pip install --user pillow
	@ pip install --user lxml
	@ pip install --user jupyter
	@ pip install --user matplotlib

# protobuf compilation
# run this in docker env. $ sudo apt-get install protobuf-compiler
protobuf_compilation:
	@ cd $(TF_RESEARCH_BASE) && protoc object_detection/protos/*.proto --python_out=./

# run this in docker env.
test_installation:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_RESEARCH_BASE)/object_detection/builders/model_builder_test.py

# download Oxford-IIIT Pets dataset
download_pets:
	@ mkdir -p datasets/pets
	@ cd datasets/pets && wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
	@ cd datasets/pets && wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
	@ cd datasets/pets && tar zxvf images.tar.gz
	@ cd datasets/pets && tar zxvf annotations.tar.gz
