# float model
QUANTOR_RESNET_V1_34_TARGETS := freeze_resnet_v1_34
QUANTOR_RESNET_V1_34_TARGETS += eval_resnet_v1_34_frozen
# float model
QUANTOR_RESNET_V1_34_TARGETS += toco_resnet_v1_34
QUANTOR_RESNET_V1_34_TARGETS += eval_resnet_v1_34_tflite
# uint8 model
QUANTOR_RESNET_V1_34_TARGETS += quantor_resnet_v1_34_frozen
QUANTOR_RESNET_V1_34_TARGETS += toco_quantor_resnet_v1_34
QUANTOR_RESNET_V1_34_TARGETS += eval_quantor_resnet_v1_34_tflite

.PHONY: train_resnet_v1_34 eval_resnet_v1_34
.PHONY: ${QUANTOR_RESNET_V1_34_TARGETS}
.PHONY: quantor_resnet_v1_34
.PHONY: compare_toco_resnet_v1_34_float compare_toco_resnet_v1_34_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for resnet_v1_34
########################################################
TF_RESNET_BASE := $(TFLITE_ROOT_PATH)/models/official/resnet
TF_MODELS_BASE := $(TFLITE_ROOT_PATH)/models

# inference model can be found in 

# resnet_v1_34_stage_0.tar.gz will be placed in /proj/mtk06790/shared/models/quantor
train_resnet_v1_34:
	@ PYTHONPATH=${TF_MODELS_BASE} \
	  python $(TF_RESNET_BASE)/imagenet_main.py --data_dir=$(DATASET_BASE)/imagenet \
		--resnet_size=34 --version 1 --model_dir=./resnet_v1_34

quantor_resnet_v1_34: ${QUANTOR_RESNET_V1_34_TARGETS}

# sub targets
freeze_resnet_v1_34:
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/resnet_v1_34/inf_graph.pbtxt \
		--input_checkpoint=$(QUANTOR_BASE)/resnet_v1_34/model.ckpt-404253 \
		--input_binary=false --output_graph=$(QUANTOR_BASE)/resnet_v1_34/frozen.pb \
		--output_node_names=softmax_tensor
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/optimize_for_inference \
		--input=$(QUANTOR_BASE)/resnet_v1_34/frozen.pb \
		--output=$(QUANTOR_BASE)/resnet_v1_34/frozen_inf.pb \
		--frozen_graph true \
		--input_names=IteratorGetNext \
		--output_names=softmax_tensor 
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v1_34/frozen_inf.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TENSORFLOW_GRAPHDEF \
		--output_file=$(QUANTOR_BASE)/resnet_v1_34/frozen_toco.pb \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=IteratorGetNext \
		--output_arrays=softmax_tensor --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet_v1_34/quantor/dots
	@ save_summaries $(QUANTOR_BASE)/resnet_v1_34/frozen_toco.pb

# Use IteratorGetNext will not work, FIXME
eval_resnet_v1_34_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=softmax_tensor \
		--input_size=224 --preprocess_name=vgg \
		--input_node_name=IteratorGetNext_1 \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_34/frozen_toco.pb \
		--max_num_batches=200  --batch_size=1

quantor_resnet_v1_34_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_34/frozen_toco.pb \
		--output_node_name=softmax_tensor \
		--input_node_name=IteratorGetNext \
		--input_size=224 --preprocess_name=vgg \
		--output_dir=$(QUANTOR_BASE)/resnet_v1_34/quantor --max_num_batches=200 --batch_size=1
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/resnet_v1_34/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/resnet_v1_34/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/resnet_v1_34/quantor/frozen.pb \
		--output_node_names=softmax_tensor
	@ save_summaries $(QUANTOR_BASE)/resnet_v1_34/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=softmax_tensor \
		--input_node_name=IteratorGetNext_1 \
		--input_size=224 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_34/quantor/frozen.pb --max_num_batches=200 --batch_size=1

toco_quantor_resnet_v1_34:
	@ mkdir -p $(QUANTOR_BASE)/resnet_v1_34/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v1_34/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet_v1_34/quantor/model.lite \
		--mean_values=114.8 --std_values=1.0 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=IteratorGetNext \
		--output_arrays=softmax_tensor --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet_v1_34/quantor/dots

toco_resnet_v1_34:
	@ mkdir -p $(QUANTOR_BASE)/resnet_v1_34/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v1_34/frozen_opt.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet_v1_34/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=IteratorGetNext \
		--output_arrays=softmax_tensor --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet_v1_34/dots

eval_quantor_resnet_v1_34_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/resnet_v1_34/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet_v1_34/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=vgg \
		--max_num_batches=2 --input_size=224

eval_resnet_v1_34_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/resnet_v1_34/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet_v1_34/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=vgg \
		--max_num_batches=2 --input_size=224


########################################################
# compare_toco
########################################################
compare_toco_resnet_v1_34_float:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_34/frozen_resnet_v1_34.pb \
		--max_num_batches=1000 \
		--output_node_name=resnet_v1_34/predictions/Reshape_1 \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=224 \
		--labels_offset=1 --preprocess_name=vgg \
		--dump_data=False

compare_toco_resnet_v1_34_uint8:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_34/quantor/frozen.pb \
		--max_num_batches=1000 \
		--tensorflow_dir=$(TF_BASE) \
		--output_node_name=resnet_v1_34/predictions/Reshape_1 \
		--toco_inference_type=uint8 \
		--input_size=224 \
		--labels_offset=1 --preprocess_name=vgg \
		--dump_data=False \
		--extra_toco_flags='--mean_values=114.8 --std_values=1.0'
