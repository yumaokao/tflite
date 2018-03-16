# float model
QUANTOR_INCEPTIONV1_TARGETS := freeze_inception_v1
QUANTOR_INCEPTIONV1_TARGETS += eval_inception_v1_frozen
# float model
QUANTOR_INCEPTIONV1_TARGETS += toco_inception_v1
QUANTOR_INCEPTIONV1_TARGETS += eval_inception_v1_tflite
# uint8 model
QUANTOR_INCEPTIONV1_TARGETS += quantor_inception_v1_frozen
QUANTOR_INCEPTIONV1_TARGETS += toco_quantor_inception_v1
QUANTOR_INCEPTIONV1_TARGETS += eval_quantor_inception_v1_tflite

.PHONY: download_inception_v1 eval_inception_v1
.PHONY: ${QUANTOR_INCEPTIONV1_TARGETS}
.PHONY: quantor_inception_v1
.PHONY: compare_toco_inception_v1_float compare_toco_inception_v1_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for inception_v1
########################################################
download_inception_v1:
	@ wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz -P $(QUANTOR_BASE)/inception_v1
	@ tar xvf $(QUANTOR_BASE)/inception_v1/inception_v1_2016_08_28.tar.gz -C $(QUANTOR_BASE)/inception_v1

quantor_inception_v1: ${QUANTOR_INCEPTIONV1_TARGETS}

# sub targets
freeze_inception_v1:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=inception_v1 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/inception_v1/inception_v1_inf_graph.pb
	@ save_summaries $(QUANTOR_BASE)/inception_v1/inception_v1_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inception_v1/inception_v1_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inception_v1/inception_v1.ckpt \
		--checkpoint_version=1 \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inception_v1/frozen_inception_v1.pb \
		--output_node_names=InceptionV1/Logits/Predictions/Reshape
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/inception_v1/frozen_inception_v1.pb
	@ save_summaries $(QUANTOR_BASE)/inception_v1/frozen_inception_v1.pb

eval_inception_v1_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV1/Logits/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/inception_v1/frozen_inception_v1.pb --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/inception_v1/summary/$@

quantor_inception_v1_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_v1/frozen_inception_v1.pb \
		--output_node_name=InceptionV1/Logits/Predictions/Reshape \
		--input_size=224 \
		--output_dir=$(QUANTOR_BASE)/inception_v1/quantor --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/inception_v1/summary/$@
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inception_v1/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inception_v1/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inception_v1/quantor/frozen.pb \
		--output_node_names=InceptionV1/Logits/Predictions/Reshape
	@ save_summaries $(QUANTOR_BASE)/inception_v1/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV1/Logits/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/inception_v1/quantor/frozen.pb --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/inception_v1/quantor/summary/$@

toco_quantor_inception_v1:
	@ mkdir -p $(QUANTOR_BASE)/inception_v1/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inception_v1/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inception_v1/quantor/model.lite \
		--mean_values=128 --std_values=127 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=InceptionV1/Logits/Predictions/Reshape --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/inception_v1/quantor/dots

toco_inception_v1:
	@ mkdir -p $(QUANTOR_BASE)/inception_v1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inception_v1/frozen_inception_v1.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inception_v1/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=InceptionV1/Logits/Predictions/Reshape --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/inception_v1/dots

eval_quantor_inception_v1_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/inception_v1/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inception_v1/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=224 --batch_size=10

eval_inception_v1_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/inception_v1/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inception_v1/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=224


########################################################
# compare_toco
########################################################
compare_toco_inception_v1_float:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_v1/frozen_inception_v1.pb \
		--max_num_batches=100 \
		--output_node_name=InceptionV1/Logits/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=224 \
		--dump_data=False

compare_toco_inception_v1_uint8:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_v1/quantor/frozen.pb \
		--max_num_batches=100 \
		--output_node_name=InceptionV1/Logits/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=uint8 \
		--input_size=224 \
		--dump_data=False
