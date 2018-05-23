.PHONY: all clean build
.PHONY: pip_install_dep coco_api_installation protobuf_compilation test_installation

.PHONY: download_mscoco
.PHONY: train_mobilenet_ssd_mscoco eval_mobilenet_ssd_mscoco
.PHONY: quant_train_mobilenet_ssd_mscoco quant_eval_mobilenet_ssd_mscoco
.PHONY: quant_eval_mobilenet_ssd_mscoco_anchors
.PHONY: quant_export_mobilenet_ssd_mscoco quant_toco_mobilenet_ssd_mscoco
.PHONY: quant_eval_mobilenet_ssd_mscoco_anchors_tflite

.PHONY: download_pretrained_mscoco
.PHONY: pretrained_eval_mobilenet_ssd_mscoco

.PHONY: download_pretrained_mobilenet
.PHONY: quant_finetune_mobilenet_ssd_mscoco

.PHONY: download_pets prepare_pets
.PHONY: train_mobilenet_ssd_pets eval_mobilenet_ssd_pets eval_mobilenet_ssd_pets_anchors
.PHONY: export_mobilenet_ssd_pets toco_mobilenet_ssd_pets
.PHONY: eval_mobilenet_ssd_pets_anchors_tflite
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


#################################
# object detection version	#
#################################
# tensorflow/models
#   * commit ad7755c81fa5bf9f9731e6cd7196f10cc30b38b8 (HEAD -> master, origin/master, origin/HEAD)
#   | Author: Vered Shwartz <vered1986@gmail.com>
#   | Date:   Thu Apr 12 08:00:28 2018 +0300
#   |
#   |     Add link to the paper and the Tratz dataset (#3745)


#################################
# dataset mscoco		#
#################################
download_mscoco:
	@ mkdir -p datasets/mscoco
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		bash download_and_preprocess_mscoco.sh datasets/mscoco $(TF_RESEARCH_BASE)
	@ cd datasets/mscoco && cp $(TF_OBJDECT_BASE)/data/mscoco_label_map.pbtxt	./

#################################
# pretrained float mscoco	#
#################################
download_pretrained_mscoco:
	@ mkdir -p mobilenet_ssd_coco_pretrained
	@ cd mobilenet_ssd_coco_pretrained \
		&& wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
	@ cd mobilenet_ssd_coco_pretrained \
		tar zxvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
	@ cd mobilenet_ssd_coco_pretrained \
		mv ssd_mobilenet_v1_coco_2017_11_17 train

# run this in docker env.
# PascalBoxes_Precision/mAP@0.5IOU:
pretrained_eval_mobilenet_ssd_mscoco:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/eval.py \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_coco_pretrained/train \
			--eval_dir mobilenet_ssd_coco_pretrained/eval \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco_pretrained.config

#################################
# float mscoco			#
#################################
# run this in docker env.
train_mobilenet_ssd_mscoco:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/train.py \
			--train_dir mobilenet_ssd_coco/train \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco.config

# run this in docker env.
# PascalBoxes_Precision/mAP@0.5IOU:
eval_mobilenet_ssd_mscoco:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/eval.py \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_coco/train \
			--eval_dir mobilenet_ssd_coco/eval \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco.config

#################################
# uint8 mscoco with mobilenet	#
#################################
download_pretrained_mobilenet:
	@ mkdir -p mobilenet_ssd_mscoco_quant_pretrained_mobilenet
	@ echo "Download MobileNet from SWRD and untar it"
	@ echo "scp mtkslt:/proj/mtk06790/shared/models/quantor-train/train_mobilenet_v1_quantize.tar.gz ./"

quant_finetune_mobilenet_ssd_mscoco:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./train.py \
			--quantize \
			--train_dir mobilenet_ssd_mscoco_quant_pretrained_mobilenet/train \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco_pretrained_mobilenet.config

#################################
# uint8 mscoco			#
#################################
# run this in docker env.
quant_train_mobilenet_ssd_mscoco:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./train.py \
			--quantize \
			--train_dir mobilenet_ssd_coco_quant/train \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco.config

# original float eval
# model.ckpt-1023685: PascalBoxes_Precision/mAP@0.5IOU: 0.222419
# run this in docker env.
quant_eval_mobilenet_ssd_mscoco:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_coco_quant/train \
			--eval_dir mobilenet_ssd_coco_quant/eval \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco.config

# model.ckpt-1023685: PascalBoxes_Precision/mAP@0.5IOU: 0.222867
# run this in docker env.
quant_eval_mobilenet_ssd_mscoco_anchors:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--evaluate_with_anchors \
			--quantize \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_coco_quant/train \
			--eval_dir mobilenet_ssd_coco_quant/eval_anchors \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco.config

# run this in docker env.
quant_export_mobilenet_ssd_mscoco:
	@ rm -rf mobilenet_ssd_coco_quant/export
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./export_inference_graph.py \
			--quantize \
			--input_type image_tensor \
			--output_directory mobilenet_ssd_coco_quant/export \
			--trained_checkpoint_prefix mobilenet_ssd_coco_quant/train/model.ckpt-1023685 \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco.config
	@ save_summaries mobilenet_ssd_coco_quant/export/frozen_inference_graph.pb
	@ $(TF_BASE)/bazel-bin/tensorflow/python/tools/optimize_for_inference \
		--input=mobilenet_ssd_coco_quant/export/frozen_inference_graph.pb \
		--output=mobilenet_ssd_coco_quant/export/optimized_inference_graph.pb \
		--input_names=Preprocessor/sub \
		--output_names=concat_1,Squeeze
	@ save_summaries mobilenet_ssd_coco_quant/export/optimized_inference_graph.pb

# run this in docker env.
quant_toco_mobilenet_ssd_mscoco:
	@ mkdir -p mobilenet_ssd_coco_quant/export/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
                --input_file=mobilenet_ssd_coco_quant/export/optimized_inference_graph.pb \
                --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
                --output_file=mobilenet_ssd_coco_quant/export//uint8_model.lite \
                --inference_type=QUANTIZED_UINT8 --inference_input_type=QUANTIZED_UINT8 \
		--input_arrays=Preprocessor/sub --mean_values=128 --std_values=127 \
                --output_arrays=concat_1,Squeeze --input_shapes=1,300,300,3 \
                --dump_graphviz=mobilenet_ssd_coco_quant/export/dots

# model.ckpt-1023685: PascalBoxes_Precision/mAP@0.5IOU: 0.220696
# run this in docker env.
quant_eval_mobilenet_ssd_mscoco_anchors_tflite:
	@ mkdir -p mobilenet_ssd_coco_quant/eval_anchors_tflite/run_tflite
	@ cp mobilenet_ssd_coco_quant/export/uint8_model.lite mobilenet_ssd_coco_quant/eval_anchors_tflite/run_tflite
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--quantize \
			--tflite_outputs box_encodings,Squeeze,0.0869676,192,class_predictions_with_background,concat_1,0.183336,214 \
			--evaluate_with_anchors \
			--evaluate_with_run_tflite \
			--tensorflow_dir $(TF_BASE) \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_coco_quant/train \
			--eval_dir mobilenet_ssd_coco_quant/eval_anchors_tflite \
			--pipeline_config_path configs/ssd_mobilenet_v1_coco.config


#################################
# dataset pets			#
#################################
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

#################################
# float pets			#
#################################
# run this in docker env.
train_mobilenet_ssd_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/train.py \
			--train_dir mobilenet_ssd_pets/train \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# run this in docker env.
# PascalBoxes_Precision/mAP@0.5IOU: @ckpt-371443 78.29%
eval_mobilenet_ssd_pets:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python $(TF_OBJDECT_BASE)/eval.py \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_pets/train \
			--eval_dir mobilenet_ssd_pets/eval \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# run this in docker env.
# PascalBoxes_Precision/mAP@0.5IOU: 0.783982
eval_mobilenet_ssd_pets_anchors:
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--evaluate_with_anchors \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_pets/train \
			--eval_dir mobilenet_ssd_pets/eval_anchors \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

# run this in docker env.
export_mobilenet_ssd_pets:
	@ rm -rf mobilenet_ssd_pets/export
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./export_inference_graph.py \
			--input_type image_tensor \
			--output_directory mobilenet_ssd_pets/export \
			--trained_checkpoint_prefix mobilenet_ssd_pets/train/model.ckpt-371443 \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config
	@ save_summaries mobilenet_ssd_pets/export/frozen_inference_graph.pb
	@ $(TF_BASE)/bazel-bin/tensorflow/python/tools/optimize_for_inference \
		--input=mobilenet_ssd_pets/export/frozen_inference_graph.pb \
		--output=mobilenet_ssd_pets/export/optimized_inference_graph.pb \
		--input_names=Preprocessor/sub \
		--output_names=concat_1,Squeeze
	@ save_summaries mobilenet_ssd_pets/export/optimized_inference_graph.pb

# run this in docker env.
toco_mobilenet_ssd_pets:
	@ mkdir -p mobilenet_ssd_pets/export/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
                --input_file=mobilenet_ssd_pets/export/optimized_inference_graph.pb \
                --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
                --output_file=mobilenet_ssd_pets/export/float_model.lite \
                --inference_type=FLOAT \
                --inference_input_type=FLOAT --input_arrays=Preprocessor/sub \
                --output_arrays=concat_1,Squeeze --input_shapes=1,300,300,3 \
                --dump_graphviz=mobilenet_ssd_pets/export/dots

# PascalBoxes_Precision/mAP@0.5IOU: 0.783982
# run this in docker env.
eval_mobilenet_ssd_pets_anchors_tflite:
	@ mkdir -p mobilenet_ssd_pets/eval_anchors_tflite/run_tflite
	@ cp mobilenet_ssd_pets/export/float_model.lite mobilenet_ssd_pets/eval_anchors_tflite/run_tflite
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--evaluate_with_anchors \
			--evaluate_with_run_tflite \
			--tflite_outputs box_encodings,Squeeze,0.0692892,154,class_predictions_with_background,concat_1,0.374646,153 \
			--tensorflow_dir $(TF_BASE) \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_pets/train \
			--eval_dir mobilenet_ssd_pets/eval_anchors_tflite \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config

#################################
# uint8 pets			#
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

# model.ckpt-351175: PascalBoxes_Precision/mAP@0.5IOU: 0.761788
# run this in docker env.
quant_eval_mobilenet_ssd_pets_anchors_tflite:
	@ mkdir -p mobilenet_ssd_pets_quant/eval_anchors_tflite/run_tflite
	@ cp mobilenet_ssd_pets_quant/export/uint8_model.lite mobilenet_ssd_pets_quant/eval_anchors_tflite/run_tflite
	@ PYTHONPATH=$(TF_RESEARCH_BASE):$(TF_SLIM_BASE) \
		python ./eval.py \
			--quantize \
			--tflite_outputs box_encodings,Squeeze,0.0692892,154,class_predictions_with_background,concat_1,0.374646,153 \
			--evaluate_with_anchors \
			--evaluate_with_run_tflite \
			--tensorflow_dir $(TF_BASE) \
			--run_once \
			--logtostderr \
			--checkpoint_dir mobilenet_ssd_pets_quant/train \
			--eval_dir mobilenet_ssd_pets_quant/eval_anchors_tflite \
			--pipeline_config_path configs/ssd_mobilenet_v1_pets.config
