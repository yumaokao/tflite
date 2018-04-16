# float model
QUANTOR_RESNET_v2_50_TARGETS := freeze_resnet_v2_50
QUANTOR_RESNET_v2_50_TARGETS += eval_resnet_v2_50_frozen
# float model
QUANTOR_RESNET_v2_50_TARGETS += toco_resnet_v2_50
QUANTOR_RESNET_v2_50_TARGETS += eval_resnet_v2_50_tflite
# uint8 model
QUANTOR_RESNET_v2_50_TARGETS += quantor_resnet_v2_50_frozen
QUANTOR_RESNET_v2_50_TARGETS += toco_quantor_resnet_v2_50
QUANTOR_RESNET_v2_50_TARGETS += eval_quantor_resnet_v2_50_tflite

.PHONY: download_resnet_v2_50 eval_resnet_v2_50
.PHONY: ${QUANTOR_RESNET_v2_50_TARGETS}
.PHONY: quantor_resnet_v2_50
.PHONY: compare_toco_resnet_v2_50_float compare_toco_resnet_v2_50_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for resnet_v2_50
########################################################
download_resnet_v2_50:
	@ wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz -P $(QUANTOR_BASE)/resnet_v2_50
	@ tar xvf $(QUANTOR_BASE)/resnet_v2_50/resnet_v2_50_2017_04_14.tar.gz -C $(QUANTOR_BASE)/resnet_v2_50

eval_resnet_v2_50:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/resnet_v2_50/resnet_v2_50.ckpt \
		--eval_dir=$(QUANTOR_BASE)/resnet_v2_50 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--preprocessing_name=inception \
		--eval_image_size=299 \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=resnet_v2_50 --max_num_batches=200

quantor_resnet_v2_50: ${QUANTOR_RESNET_v2_50_TARGETS}

# sub targets
freeze_resnet_v2_50:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=resnet_v2_50 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/resnet_v2_50/resnet_v2_50_inf_graph.pb
	@ save_summaries $(QUANTOR_BASE)/resnet_v2_50/resnet_v2_50_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/resnet_v2_50/resnet_v2_50_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/resnet_v2_50/resnet_v2_50.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/resnet_v2_50/frozen_resnet_v2_50.pb \
		--output_node_names=resnet_v2_50/predictions/Reshape_1
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v2_50/frozen_resnet_v2_50.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TENSORFLOW_GRAPHDEF \
		--output_file=$(QUANTOR_BASE)/resnet_v2_50/frozen_toco.pb \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=resnet_v2_50/predictions/Reshape_1 --input_shapes=1,224,224,3
	@ save_summaries $(QUANTOR_BASE)/resnet_v2_50/frozen_resnet_v2_50.pb

eval_resnet_v2_50_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=resnet_v2_50/predictions/Reshape_1 \
		--input_size=224 --preprocess_name=inception \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v2_50/frozen_resnet_v2_50.pb --max_num_batches=200

quantor_resnet_v2_50_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v2_50/frozen_toco.pb \
		--output_node_name=resnet_v2_50/predictions/Reshape_1 \
		--input_size=224 --preprocess_name=inception \
		--output_dir=$(QUANTOR_BASE)/resnet_v2_50/quantor \
		--max_num_batches=200
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/resnet_v2_50/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/resnet_v2_50/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/resnet_v2_50/quantor/frozen.pb \
		--output_node_names=resnet_v2_50/predictions/Reshape_1
	@ save_summaries $(QUANTOR_BASE)/resnet_v2_50/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=resnet_v2_50/predictions/Reshape_1 \
		--input_size=224 --preprocess_name=inception \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v2_50/quantor/frozen.pb \
		--max_num_batches=200

toco_quantor_resnet_v2_50:
	@ mkdir -p $(QUANTOR_BASE)/resnet_v2_50/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v2_50/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet_v2_50/quantor/model.lite \
		--mean_values=128 --std_values=127 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=resnet_v2_50/predictions/Reshape_1 --input_shapes=10,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet_v2_50/quantor/dots

toco_resnet_v2_50:
	@ mkdir -p $(QUANTOR_BASE)/resnet_v2_50/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v2_50/frozen_resnet_v2_50.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet_v2_50/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=resnet_v2_50/predictions/Reshape_1 --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet_v2_50/dots

eval_quantor_resnet_v2_50_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/resnet_v2_50/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet_v2_50/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=inception \
		--max_num_batches=1000 --input_size=224 --batch_size=10

eval_resnet_v2_50_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/resnet_v2_50/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet_v2_50/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=inception \
		--max_num_batches=10000 --input_size=224


########################################################
# compare_toco
########################################################
compare_toco_resnet_v2_50_float:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v2_50/frozen_toco.pb \
		--max_num_batches=1000 \
		--output_node_name=resnet_v2_50/predictions/Reshape_1 \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=224 \
		--preprocess_name=inception \
		--dump_data=False

compare_toco_resnet_v2_50_uint8:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v2_50/quantor/frozen.pb \
		--max_num_batches=1000 \
		--tensorflow_dir=$(TF_BASE) \
		--output_node_name=resnet_v2_50/predictions/Reshape_1 \
		--toco_inference_type=uint8 \
		--input_size=224 \
		--preprocess_name=inception \
		--dump_data=False \
		--extra_toco_flags='--mean_values=114.8 --std_values=1.0'
