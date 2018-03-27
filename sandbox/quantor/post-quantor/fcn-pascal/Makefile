.PHONY: all clean build
.PHONY: download_dataset prepare_dataset convert_tfrecords
.PHONY: download_pretrained_vgg16
.PHONY: clone_tf_image_segmentation pip_install_requirements
.PHONY: train_fcn_32s train_fcn_16s train_fcn_8s
.PHONY: test_fcn_32s test_fcn_16s test_fcn_8s
.PHONY: tar_fcn_32s tar_fcn_16s tar_fcn_8s
.PHONY: export_fcn_32s toco_fcn_32s test_fcn_32s_tflite
.PHONY: gen_samples_fcn_32s

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

clone_tf_image_segmentation:
	# @ git clone https://github.com/warmspringwinds/tf-image-segmentation.git
	@ git clone https://github.com/barryridge/tf-image-segmentation.git

# TODO: makefile 
# tf-image-implementaion needs patch to work with newer tensorflow (1.6 tested)

# fcn_32s:
# (RV-VOC12) Mean IU: 0.6182407
train_fcn_32s:
	@ python fcn_32s/fcn_32s_train.py
test_fcn_32s:
	@ python fcn_32s/fcn_32s_test_pascal.py --slim_dir=$(TF_SLIM_BASE)
export_fcn_32s:
	@ mkdir -p fcn_32s/models
	@ python fcn_32s/fcn_32s_export.py --slim_dir=$(TF_SLIM_BASE)
toco_fcn_32s:
	@ mkdir -p fcn_32s/models/dots
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
test_fcn_32s_tflite:
	@ python fcn_32s/fcn_32s_tflite_float.py \
		--tensorflow_dir=$(TF_BASE) \
		--tflite_model=$(FCN_BASE)/fcn_32s/models/FCN_float.lite \
		--slim_dir=$(TF_SLIM_BASE)
tar_fcn_32s:
	@ tar zcvf fcn_32s.tar.gz fcn_32s
gen_samples_fcn_32s:
	@ python fcn_32s/fcn_32s_test_pascal_samples.py

# fcn_16s:
# (RV-VOC12) Mean IU: 0.63535184
train_fcn_16s:
	@ python fcn_16s/fcn_16s_train.py
test_fcn_16s:
	@ python fcn_16s/fcn_16s_test_pascal.py
tar_fcn_16s:
	@ tar zcvf fcn_16s.tar.gz fcn_16s

# fcn_8s:
# (RV-VOC12) Mean IU: 0.6364892
train_fcn_8s:
	@ python fcn_8s/fcn_8s_train.py
test_fcn_8s:
	@ python fcn_8s/fcn_8s_test_pascal.py
tar_fcn_8s:
	@ tar zcvf fcn_8s.tar.gz fcn_8s

# *.tar.gz will be placed in /proj/mtk06790/shared/models/fcn-pascal/
