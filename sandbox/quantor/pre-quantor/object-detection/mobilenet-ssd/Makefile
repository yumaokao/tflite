.PHONY: all clean build
.PHONY: pip_install_dep coco_api_installation protobuf_compilation test_installation
.PHONY: download_pets prepare_pets
.PHONY: train_mobilenet_ssd_pets eval_mobilenet_ssd_pets

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
	@ sudo apt-get install protobuf-compiler
	@ sudo apt-get install python-tk
	@ pip install --user Cython
	@ pip install --user pillow
	@ pip install --user lxml
	@ pip install --user jupyter
	@ pip install --user matplotlib

# coco api installation
# run this in docker env.
coco_api_installation:
	@ git clone https://github.com/cocodataset/cocoapi.git
	@ cd cocoapi/PythonAPI && make
	@ cd cocoapi/PythonAPI && cp -r pycocotools $(TF_RESEARCH_BASE)

# protobuf compilation
# run this in docker env. $ sudo apt-get install protobuf-compiler
protobuf_compilation:
	@ cd $(TF_RESEARCH_BASE) && protoc object_detection/protos/*.proto --python_out=./

# run this in docker env.
test_installation:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_RESEARCH_BASE)/object_detection/builders/model_builder_test.py

# download and prepare Oxford-IIIT Pets dataset
download_pets:
	@ mkdir -p datasets/pets
	@ cd datasets/pets && wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
	@ cd datasets/pets && wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
	@ cd datasets/pets && tar zxvf images.tar.gz
	@ cd datasets/pets && tar zxvf annotations.tar.gz
	@ cd datasets/pets && $(TF_OBJDECT_BASE)/data/pet_label_map.pbtxt ./

# run this in docker env.
prepare_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_RESEARCH_BASE)/object_detection/dataset_tools/create_pet_tf_record.py \
			-label_map_path=$(TF_RESEARCH_BASE)/object_detection/data/pet_label_map.pbtxt \
			--data_dir=datasets/pets --output_dir=datasets/pets

# run this in docker env.
train_mobilenet_ssd_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/train.py \
			--train_dir train_mobilenet_ssd_pets \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# run this in docker env.
# needed to view mAP in tensorboard
#     PascalBoxes_Precision/mAP@0.5IOU: @ckpt-371443 78.29%
eval_mobilenet_ssd_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/eval.py \
			--logtostderr \
			--checkpoint_dir train_mobilenet_ssd_pets \
			--eval_dir eval_mobilenet_ssd_pets \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config
