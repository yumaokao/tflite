.PHONY: all clean build
.PHONY: download_dataset prepare_dataset convert_tfrecords
.PHONY: download_pretrained_vgg16
.PHONY: clone_tf_image_segmentation pip_install_requirements
.PHONY: train_fcn_32s train_fcn_16s train_fcn_8s
.PHONY: test_fcn_32s test_fcn_16s test_fcn_8s
.PHONY: tar_fcn_32s tar_fcn_16s tar_fcn_8s
.PHONY: export_fcn_32s toco_fcn_32s test_fcn_32s_tflite
.PHONY: gen_samples_fcn_32s
# tflite fcn_32s
TFLITE_FCN_32S_TARGETS := export_fcn_32s
TFLITE_FCN_32S_TARGETS += toco_fcn_32s
TFLITE_FCN_32S_TARGETS += test_fcn_32s_tflite
TFLITE_FCN_32S_TARGETS += quantor_fcn_32s
TFLITE_FCN_32S_TARGETS += toco_quantor_fcn_32s
TFLITE_FCN_32S_TARGETS += test_quantor_fcn_32s_tflite
.PHONY: $(TFLITE_FCN_32S_TARGETS)
# tflite fcn_16s
TFLITE_FCN_16S_TARGETS := export_fcn_16s
TFLITE_FCN_16S_TARGETS += toco_fcn_16s
TFLITE_FCN_16S_TARGETS += test_fcn_16s_tflite
TFLITE_FCN_16S_TARGETS += quantor_fcn_16s
TFLITE_FCN_16S_TARGETS += toco_quantor_fcn_16s
TFLITE_FCN_16S_TARGETS += test_quantor_fcn_16s_tflite
.PHONY: $(TFLITE_FCN_16S_TARGETS)

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
FCN_BASE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

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

# The fcn implementaion is derived from https://github.com/warmspringwinds/tf-image-segmentation
# But clone https://github.com/barryridge/tf-image-segmentation.git for newer tensorflow support
# Download and git add back, to modified for quantization

# in docker with tensorflow
pip_install_requirements:
	@ pip install --user scikit-image

pip_install_requirements_venv:
	@ pip install scikit-image

# dataset.tar.gz will be placed in /proj/mtk06790/shared/models/fcn-pascal/
download_dataset:
	@ mkdir -p datasets
	@ cd datasets && wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
	@ cd datasets && wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

prepare_dataset:
	@ cd datasets && tar xvf VOCtrainval_11-May-2012.tar
	@ cd datasets && tar zxvf benchmark.tgz

# in docker with tensorflow
convert_tfrecords:
	@ python datasets/convert_pascal_berkeley_augmented_mat_annotations_to_png.py
	@ python datasets/convert_pascal_voc_tfrecords.py

download_pretrained_vgg16:
	@ mkdir -p vgg_16_ckpts
	@ cd vgg_16_ckpts && wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
	@ cd vgg_16_ckpts && tar zxvf vgg_16_2016_08_28.tar.gz

# No need to since already have one in repo, for reference only
# tf-image-implementaion-upstream needs patch to work with newer tensorflow (1.6 tested)
# See patches
clone_tf_image_segmentation:
	# @ git clone https://github.com/warmspringwinds/tf-image-segmentation.git
	@ git clone https://github.com/barryridge/tf-image-segmentation.git tf-image-segmentation-upstream

# ######################################
# fcn_32s:
# (RV-VOC12) Mean IU: 0.6182407
# ######################################
train_fcn_32s:
	@ python fcn_32s/fcn_32s_train.py
test_fcn_32s:
	@ python fcn_32s/fcn_32s_test_pascal.py --slim_dir=$(TF_SLIM_BASE)
tar_fcn_32s:
	@ tar zcvf fcn_32s.tar.gz fcn_32s
gen_samples_fcn_32s:
	@ python fcn_32s/fcn_32s_test_pascal_samples.py

tflite_fcn_32s: $(TFLITE_FCN_32S_TARGETS)

export_fcn_32s:
	@ mkdir -p $(FCN_BASE)/fcn_32s/models
	@ python fcn_32s/fcn_32s_export.py --slim_dir=$(TF_SLIM_BASE)
toco_fcn_32s:
	@ mkdir -p $(FCN_BASE)/fcn_32s/models/dots
	@ cd $(TF_BASE) && bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(FCN_BASE)/fcn_32s/models/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=$(FCN_BASE)/fcn_32s/models/FCN_float.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=fcn_32s/sub \
		--output_arrays=fcn_32s/prediction \
		--input_shapes=1,384,512,3 \
		--dump_graphviz=$(FCN_BASE)/fcn_32s/models/dots
# (RV-VOC12) Mean IU: 0.6150873(405 val images)
test_fcn_32s_tflite:
	@ echo $@
	@ python fcn_32s/fcn_32s_tflite.py \
		--tensorflow_dir=$(TF_BASE) \
		--tflite_model=$(FCN_BASE)/fcn_32s/models/FCN_float.lite \
		--inference_type=float \
		--output_node_name=fcn_32s/prediction \
		--slim_dir=$(TF_SLIM_BASE)
# (RV-VOC12) Mean IU: 0.6137654(405 images)
quantor_fcn_32s:
	@ mkdir -p $(FCN_BASE)/fcn_32s/models/quantor
	@ python fcn_32s/fcn_32s_quantor.py \
		--frozen_pb=$(FCN_BASE)/fcn_32s/models/frozen.pb \
		--input_node_name=fcn_32s/sub \
		--output_node_name=fcn_32s/act_quant/FakeQuantWithMinMaxVars \
		--output_dir=$(FCN_BASE)/fcn_32s/models/quantor \
		--slim_dir=$(TF_SLIM_BASE) \
		--num_batches=1000 \
		--summary_dir=$(FCN_BASE)/fcn_32s/models/quantor/summary
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(FCN_BASE)/fcn_32s/models/quantor/quantor.pb \
		--input_checkpoint=$(FCN_BASE)/fcn_32s/models/quantor/model.ckpt \
		--input_binary=true --output_graph=$(FCN_BASE)/fcn_32s/models/quantor/frozen.pb \
		--output_node_names=fcn_32s/act_quant/FakeQuantWithMinMaxVars
	@ python fcn_32s/fcn_32s_test_frozen.py \
		--frozen_pb=$(FCN_BASE)/fcn_32s/models/quantor/frozen.pb \
		--input_node_name=fcn_32s/sub \
		--output_node_name=fcn_32s/act_quant/FakeQuantWithMinMaxVars \
		--output_dir=$(FCN_BASE)/fcn_32s/models/quantor \
		--slim_dir=$(TF_SLIM_BASE)
toco_quantor_fcn_32s:
	@ mkdir -p $(FCN_BASE)/fcn_32s/models/quantor/dots
	@ cd $(TF_BASE) && bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(FCN_BASE)/fcn_32s/models/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=$(FCN_BASE)/fcn_32s/models/quantor/FCN_uint8.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 \
		--input_arrays=fcn_32s/sub \
		--output_arrays=fcn_32s/act_quant/FakeQuantWithMinMaxVars \
		--mean_values=114.8 \
		--std_values=1 \
		--input_shapes=1,384,512,3 \
		--dump_graphviz=$(FCN_BASE)/fcn_32s/models/quantor/dots
# (RV-VOC12) Mean IU: 0.6153046(405 images)
test_quantor_fcn_32s_tflite:
	@ echo $@
	@ python fcn_32s/fcn_32s_tflite.py \
		--tensorflow_dir=$(TF_BASE) \
		--tflite_model=$(FCN_BASE)/fcn_32s/models/quantor/FCN_uint8.lite \
		--inference_type=uint8 \
		--output_node_name=fcn_32s/act_quant/FakeQuantWithMinMaxVars \
		--slim_dir=$(TF_SLIM_BASE)

# ######################################
# fcn_16s:
# (RV-VOC12) Mean IU: 0.63535184
# ######################################
train_fcn_16s:
	@ python fcn_16s/fcn_16s_train.py
test_fcn_16s:
	@ python fcn_16s/fcn_16s_test_pascal.py
tar_fcn_16s:
	@ tar zcvf fcn_16s.tar.gz fcn_16s

tflite_fcn_16s: $(TFLITE_FCN_16S_TARGETS)

export_fcn_16s:
	@ mkdir -p $(FCN_BASE)/fcn_16s/models
	@ python fcn_16s/fcn_16s_export.py --slim_dir=$(TF_SLIM_BASE)
toco_fcn_16s:
	@ mkdir -p $(FCN_BASE)/fcn_16s/models/dots
	@ cd $(TF_BASE) && bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(FCN_BASE)/fcn_16s/models/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=$(FCN_BASE)/fcn_16s/models/FCN_float.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=fcn_16s/sub \
		--output_arrays=fcn_16s/prediction \
		--input_shapes=1,384,512,3 \
		--dump_graphviz=$(FCN_BASE)/fcn_16s/models/dots
# (RV-VOC12) Mean IU: 0.6150873(405 val images)
test_fcn_16s_tflite:
	@ echo $@
	@ python fcn_16s/fcn_16s_tflite.py \
		--tensorflow_dir=$(TF_BASE) \
		--tflite_model=$(FCN_BASE)/fcn_16s/models/FCN_float.lite \
		--inference_type=float \
		--output_node_name=fcn_16s/prediction \
		--slim_dir=$(TF_SLIM_BASE)
# (RV-VOC12) Mean IU: 0.6137654(405 images)
quantor_fcn_16s:
	@ mkdir -p $(FCN_BASE)/fcn_16s/models/quantor
	@ python fcn_16s/fcn_16s_quantor.py \
		--frozen_pb=$(FCN_BASE)/fcn_16s/models/frozen.pb \
		--input_node_name=fcn_16s/sub \
		--output_node_name=fcn_16s/act_quant_1/FakeQuantWithMinMaxVars \
		--output_dir=$(FCN_BASE)/fcn_16s/models/quantor \
		--slim_dir=$(TF_SLIM_BASE) \
		--num_batches=1000 \
		--summary_dir=$(FCN_BASE)/fcn_16s/models/quantor/summary
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(FCN_BASE)/fcn_16s/models/quantor/quantor.pb \
		--input_checkpoint=$(FCN_BASE)/fcn_16s/models/quantor/model.ckpt \
		--input_binary=true --output_graph=$(FCN_BASE)/fcn_16s/models/quantor/frozen.pb \
		--output_node_names=fcn_16s/act_quant_1/FakeQuantWithMinMaxVars
	@ python fcn_16s/fcn_16s_test_frozen.py \
		--frozen_pb=$(FCN_BASE)/fcn_16s/models/quantor/frozen.pb \
		--input_node_name=fcn_16s/sub \
		--output_node_name=fcn_16s/act_quant_1/FakeQuantWithMinMaxVars \
		--output_dir=$(FCN_BASE)/fcn_16s/models/quantor \
		--slim_dir=$(TF_SLIM_BASE)
toco_quantor_fcn_16s:
	@ mkdir -p $(FCN_BASE)/fcn_16s/models/quantor/dots
	@ cd $(TF_BASE) && bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(FCN_BASE)/fcn_16s/models/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=$(FCN_BASE)/fcn_16s/models/quantor/FCN_uint8.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 \
		--input_arrays=fcn_16s/sub \
		--output_arrays=fcn_16s/act_quant_1/FakeQuantWithMinMaxVars \
		--mean_values=114.8 \
		--std_values=1 \
		--input_shapes=1,384,512,3 \
		--dump_graphviz=$(FCN_BASE)/fcn_16s/models/quantor/dots
# (RV-VOC12) Mean IU: 0.6153046(405 images)
test_quantor_fcn_16s_tflite:
	@ echo $@
	@ python fcn_16s/fcn_16s_tflite.py \
		--tensorflow_dir=$(TF_BASE) \
		--tflite_model=$(FCN_BASE)/fcn_16s/models/quantor/FCN_uint8.lite \
		--inference_type=uint8 \
		--output_node_name=fcn_16s/act_quant_1/FakeQuantWithMinMaxVars \
		--slim_dir=$(TF_SLIM_BASE)

# ######################################
# fcn_8s:
# (RV-VOC12) Mean IU: 0.6364892
# ######################################
train_fcn_8s:
	@ python fcn_8s/fcn_8s_train.py
test_fcn_8s:
	@ python fcn_8s/fcn_8s_test_pascal.py
tar_fcn_8s:
	@ tar zcvf fcn_8s.tar.gz fcn_8s

# *.tar.gz will be placed in /proj/mtk06790/shared/models/fcn-pascal/
