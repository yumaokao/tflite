# float model
QUANTOR_VGG_16_TARGETS := freeze_vgg_16
QUANTOR_VGG_16_TARGETS += eval_vgg_16_frozen
# float model
QUANTOR_VGG_16_TARGETS += toco_vgg_16
QUANTOR_VGG_16_TARGETS += eval_vgg_16_tflite
# uint8 model
QUANTOR_VGG_16_TARGETS += quantor_vgg_16_frozen
QUANTOR_VGG_16_TARGETS += toco_quantor_vgg_16
QUANTOR_VGG_16_TARGETS += eval_quantor_vgg_16_tflite

.PHONY: download_vgg_16 eval_vgg_16
.PHONY: ${QUANTOR_VGG_16_TARGETS}
.PHONY: quantor_vgg_16
.PHONY: compare_toco_vgg_16_float compare_toco_vgg_16_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for vgg_16
########################################################
download_vgg_16:
	@ wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -P $(QUANTOR_BASE)/vgg_16
	@ tar xvf $(QUANTOR_BASE)/vgg_16/vgg_16_2016_08_28.tar.gz -C $(QUANTOR_BASE)/vgg_16

eval_vgg_16:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/vgg_16/vgg_16.ckpt \
		--eval_dir=$(QUANTOR_BASE)/vgg_16 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--labels_offset=1 \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=vgg_16 --max_num_batches=200

quantor_vgg_16: ${QUANTOR_VGG_16_TARGETS}

# sub targets
freeze_vgg_16:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr --labels_offset=1 \
		--model_name=vgg_16 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/vgg_16/vgg_16_inf_graph.pb
	@ save_summaries $(QUANTOR_BASE)/vgg_16/vgg_16_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/vgg_16/vgg_16_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/vgg_16/vgg_16.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb \
		--output_node_names=vgg_16/fc8/squeezed
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
		--in_graph=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb \
		--out_graph=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16_tmp.pb \
		--inputs=input \
		--outputs=vgg_16/fc8/squeezed \
		--transforms='fold_old_batch_norms fold_batch_norms'
	@ mv $(QUANTOR_BASE)/vgg_16/frozen_vgg_16_tmp.pb $(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb
	@ save_summaries $(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb

convert_vgg_16_fc:
	@ python $(QUANTOR_BASE)/vgg_16_convert.py \
		--frozen_pb=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb \
		--output_pb=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16_fc.pb

eval_vgg_16_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=vgg_16/fc8/squeezed \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb --max_num_batches=200

eval_vgg_16_fc_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=vgg_16/fc8/fc/BiasAdd \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16_fc.pb --max_num_batches=200

quantor_vgg_16_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb \
		--output_node_name=vgg_16/fc8/squeezed \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--output_dir=$(QUANTOR_BASE)/vgg_16/quantor --max_num_batches=200
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/vgg_16/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/vgg_16/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/vgg_16/quantor/frozen.pb \
		--output_node_names=vgg_16/fc8/squeezed
	@ save_summaries $(QUANTOR_BASE)/vgg_16/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=vgg_16/fc8/squeezed \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/vgg_16/quantor/frozen.pb --max_num_batches=200

# --default_ranges_min=0 --default_ranges_max=10
toco_quantor_vgg_16:
	@ mkdir -p $(QUANTOR_BASE)/vgg_16/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/vgg_16/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/vgg_16/quantor/model.lite \
		--mean_values=114.8 --std_values=1.0 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=vgg_16/fc8/squeezed --input_shapes=10,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/vgg_16/quantor/dots

toco_vgg_16:
	@ mkdir -p $(QUANTOR_BASE)/vgg_16/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/vgg_16/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=vgg_16/fc8/squeezed --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/vgg_16/dots

eval_quantor_vgg_16_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/vgg_16/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/vgg_16/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--labels_offset=1 --preprocess_name=vgg \
		--max_num_batches=1000 --input_size=224 --batch_size=10

eval_vgg_16_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/vgg_16/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/vgg_16/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--labels_offset=1 --preprocess_name=vgg \
		--max_num_batches=10000 --input_size=224


########################################################
# compare_toco
########################################################
compare_toco_vgg_16_float:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/vgg_16/frozen_vgg_16.pb \
		--max_num_batches=1000 \
		--output_node_name=vgg_16/fc8/squeezed \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=224 \
		--labels_offset=1 --preprocess_name=vgg \
		--dump_data=False

compare_toco_vgg_16_uint8:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/vgg_16/quantor/frozen.pb \
		--max_num_batches=1000 \
		--tensorflow_dir=$(TF_BASE) \
		--output_node_name=vgg_16/fc8/squeezed \
		--toco_inference_type=uint8 \
		--input_size=224 \
		--labels_offset=1 --preprocess_name=vgg \
		--dump_data=False \
		--extra_toco_flags='--mean_values=114.8 --std_values=1.0'
