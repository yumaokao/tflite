# float model
QUANTOR_RESNET34_TARGETS += eval_resnet34_frozen
# float model
QUANTOR_RESNET34_TARGETS += toco_resnet34
QUANTOR_RESNET34_TARGETS += eval_resnet34_tflite
# uint8 model
QUANTOR_RESNET34_TARGETS += quantor_resnet34_frozen
QUANTOR_RESNET34_TARGETS += toco_quantor_resnet34
QUANTOR_RESNET34_TARGETS += eval_quantor_resnet34_tflite

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for resnet32
########################################################

download_resnet34:
	@ echo "Please manually download the model from \`\\NB17030014\Users\mtk07832\Desktop\shared\resnet34_frozen_pb\`."

quantor_resnet34: ${QUANTOR_RESNET34_TARGETS}

eval_resnet34_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=softmax_cross_entropy_loss/xentropy/Reshape \
		--input_size=224 \
		--preprocess_name=vgg_official \
		--frozen_pb=$(QUANTOR_BASE)/resnet34/frozen_resnet34.pb \
		--max_num_batches=1000 \
		--batch_size=1

quantor_resnet34_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet34/frozen_resnet34.pb \
		--output_node_name=softmax_cross_entropy_loss/xentropy/Reshape \
		--input_size=224 \
		--preprocess_name=vgg_official \
		--output_dir=$(QUANTOR_BASE)/resnet34/quantor \
		--max_num_batches=200 \
		--batch_size=1
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/resnet34/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/resnet34/quantor/model.ckpt \
		--input_binary=true \
		--output_graph=$(QUANTOR_BASE)/resnet34/quantor/frozen.pb \
		--output_node_names=softmax_cross_entropy_loss/xentropy/Reshape
	@ save_summaries $(QUANTOR_BASE)/resnet34/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=softmax_cross_entropy_loss/xentropy/Reshape \
		--input_size=224 \
		--preprocess_name=vgg_official \
		--frozen_pb=$(QUANTOR_BASE)/resnet34/quantor/frozen.pb \
		--max_num_batches=200 \
		--batch_size=1

toco_resnet34:
	@ mkdir -p $(QUANTOR_BASE)/resnet34/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet34/frozen_resnet34.pb \
		--input_format=TENSORFLOW_GRAPHDEF  \
		--output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet34/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=input \
		--output_arrays=softmax_cross_entropy_loss/xentropy/Reshape \
		--input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet34/dots

eval_resnet34_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/resnet34/quantor/summary/$@ \
		--dataset_name=imagenet \
		--dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet34/float_model.lite \
		--tensorflow_dir=$(TF_BASE) \
		--max_num_batches=10000 \
		--input_size=224

toco_quantor_resnet34:
	@ mkdir -p $(QUANTOR_BASE)/resnet34/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet34/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  \
		--output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet34/quantor/model.lite \
		--preprocess_name=vgg_official \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--mean_values=114.8 --std_values=255.0 \
		--output_arrays=softmax_cross_entropy_loss/xentropy/Reshape \
		--input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet34/quantor/dots

eval_quantor_resnet34_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/resnet34/quantor/summary/$@ \
		--dataset_name=imagenet \
		--dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet34/quantor/model.lite \
		--inference_type=uint8 \
		--tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 \
		--input_size=224

