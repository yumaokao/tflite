.PHONY: all clean build
.PHONY: pip_install_dep coco_api_installation protobuf_compilation test_installation
.PHONY: download_pets prepare_pets
.PHONY: train_mobilenet_ssd_pets eval_mobilenet_ssd_pets eval_mobilenet_ssd_pets_anchors
.PHONY: export_mobilenet_ssd_pets toco_mobilenet_ssd_pets
.PHONY: eval_mobilenet_ssd_pets_anchors_tflite
# quant_train_mobilenet_ssd_pets
.PHONY: quant_train_mobilenet_ssd_pets
.PHONY: quant_eval_mobilenet_ssd_pets quant_eval_mobilenet_ssd_pets_anchors
.PHONY: quant_export_mobilenet_ssd_pets quant_toco_mobilenet_ssd_pets
.PHONY: quant_eval_mobilenet_ssd_pets_anchors_tflite

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
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:optimize_for_inference
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
	@ cd datasets/pets && cp $(TF_OBJDECT_BASE)/data/pet_label_map.pbtxt ./

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
# PascalBoxes_Precision/mAP@0.5IOU: @ckpt-371443 78.29%
eval_mobilenet_ssd_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/eval.py \
			--logtostderr \
			--checkpoint_dir train_mobilenet_ssd_pets \
			--eval_dir eval_mobilenet_ssd_pets \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# run this in docker env.
# PascalBoxes_Precision/mAP@0.5IOU: 0.783982
eval_mobilenet_ssd_pets_anchors:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--evaluate_with_anchors \
			--run_once \
			--logtostderr \
			--checkpoint_dir train_mobilenet_ssd_pets \
			--eval_dir eval_mobilenet_ssd_pets_anchors \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# run this in docker env.
export_mobilenet_ssd_pets:
	@ rm -rf export_mobilenet_ssd_pets
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./export_inference_graph.py \
			--input_type image_tensor \
			--output_directory export_mobilenet_ssd_pets \
			--trained_checkpoint_prefix train_mobilenet_ssd_pets/model.ckpt-371443 \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config
	@ save_summaries export_mobilenet_ssd_pets/frozen_inference_graph.pb
	@ $(TF_BASE)/bazel-bin/tensorflow/python/tools/optimize_for_inference \
		--input=export_mobilenet_ssd_pets/frozen_inference_graph.pb \
		--output=export_mobilenet_ssd_pets/optimized_inference_graph.pb \
		--input_names=Preprocessor/sub \
		--output_names=concat_1,Squeeze
	@ save_summaries export_mobilenet_ssd_pets/optimized_inference_graph.pb


# run this in docker env.
toco_mobilenet_ssd_pets:
	@ mkdir -p export_mobilenet_ssd_pets/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
                --input_file=export_mobilenet_ssd_pets/optimized_inference_graph.pb \
                --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
                --output_file=export_mobilenet_ssd_pets/float_model.lite \
                --inference_type=FLOAT \
                --inference_input_type=FLOAT --input_arrays=Preprocessor/sub \
                --output_arrays=concat_1,Squeeze --input_shapes=1,300,300,3 \
                --dump_graphviz=export_mobilenet_ssd_pets/dots


# TODO(yumaokao): too many export_* eval_*, should move to a mobilenet_ssd_pets
# PascalBoxes_Precision/mAP@0.5IOU: 0.783982
# run this in docker env.
eval_mobilenet_ssd_pets_anchors_tflite:
	@ mkdir -p eval_mobilenet_ssd_pets_anchors_tflite/run_tflite
	@ cp export_mobilenet_ssd_pets/float_model.lite eval_mobilenet_ssd_pets_anchors_tflite/run_tflite
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--evaluate_with_anchors \
			--evaluate_with_run_tflite \
			--tensorflow_dir $(TF_BASE) \
			--run_once \
			--logtostderr \
			--checkpoint_dir train_mobilenet_ssd_pets \
			--eval_dir eval_mobilenet_ssd_pets_anchors_tflite \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

#################################
# quantized 			#
#################################
# run this in docker env.
quant_train_mobilenet_ssd_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./train.py \
			--quantize \
			--train_dir mobilenet_ssd_pets_quant/train \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# original float eval
# model.ckpt-351175: PascalBoxes_Precision/mAP@0.5IOU: 0.764092
# run this in docker env.
quant_eval_mobilenet_ssd_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_pets_quant/train \
			--eval_dir mobilenet_ssd_pets_quant/eval \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# model.ckpt-351175: PascalBoxes_Precision/mAP@0.5IOU: 0.762136
# run this in docker env.
quant_eval_mobilenet_ssd_pets_anchors:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--evaluate_with_anchors \
			--quantize \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_pets_quant/train \
			--eval_dir mobilenet_ssd_pets_quant/eval_anchors \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# run this in docker env.
quant_export_mobilenet_ssd_pets:
	@ rm -rf mobilenet_ssd_pets_quant/export
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./export_inference_graph.py \
			--quantize \
			--input_type image_tensor \
			--output_directory mobilenet_ssd_pets_quant/export \
			--trained_checkpoint_prefix mobilenet_ssd_pets_quant/train/model.ckpt-351175 \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config
	@ save_summaries mobilenet_ssd_pets_quant/export/frozen_inference_graph.pb
	@ $(TF_BASE)/bazel-bin/tensorflow/python/tools/optimize_for_inference \
		--input=mobilenet_ssd_pets_quant/export/frozen_inference_graph.pb \
		--output=mobilenet_ssd_pets_quant/export/optimized_inference_graph.pb \
		--input_names=Preprocessor/sub \
		--output_names=concat_1,Squeeze
	@ save_summaries mobilenet_ssd_pets_quant/export/optimized_inference_graph.pb

# run this in docker env.
quant_toco_mobilenet_ssd_pets:
	@ mkdir -p mobilenet_ssd_pets_quant/export/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
                --input_file=mobilenet_ssd_pets_quant/export/optimized_inference_graph.pb \
                --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
                --output_file=mobilenet_ssd_pets_quant/export//uint8_model.lite \
                --inference_type=QUANTIZED_UINT8 --inference_input_type=QUANTIZED_UINT8 \
		--input_arrays=Preprocessor/sub --mean_values=128 --std_values=127 \
                --output_arrays=concat_1,Squeeze --input_shapes=1,300,300,3 \
                --dump_graphviz=mobilenet_ssd_pets_quant/export/dots

# PascalBoxes_Precision/mAP@0.5IOU: 0.761788
# run this in docker env.
quant_eval_mobilenet_ssd_pets_anchors_tflite:
	@ mkdir -p mobilenet_ssd_pets_quant/eval_anchors_tflite/run_tflite
	@ cp mobilenet_ssd_pets_quant/export/uint8_model.lite mobilenet_ssd_pets_quant/eval_anchors_tflite/run_tflite
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--quantize \
			--evaluate_with_anchors \
			--evaluate_with_run_tflite \
			--tensorflow_dir $(TF_BASE) \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_pets_quant/train \
			--eval_dir mobilenet_ssd_pets_quant/eval_anchors_tflite \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config
