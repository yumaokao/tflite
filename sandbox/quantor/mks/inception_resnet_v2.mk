# float model
QUANTOR_INCPETIONRESNETV2_TARGETS := freeze_inception_resnet_v2
QUANTOR_INCPETIONRESNETV2_TARGETS += eval_inception_resnet_v2_frozen
# float (fake quanted) model
QUANTOR_INCPETIONRESNETV2_TARGETS += toco_inception_resnet_v2
QUANTOR_INCPETIONRESNETV2_TARGETS += eval_inception_resnet_v2_tflite
# uint8 model
QUANTOR_INCPETIONRESNETV2_TARGETS += quantor_inception_resnet_v2_frozen
QUANTOR_INCPETIONRESNETV2_TARGETS += toco_quantor_inception_resnet_v2
QUANTOR_INCPETIONRESNETV2_TARGETS += eval_quantor_inception_resnet_v2_tflite

.PHONY: download_inception_resnet_v2 eval_inception_resnet_v2
.PHONY: ${QUANTOR_INCPETIONRESNETV2_TARGETS}
.PHONY: quantor_inception_resnet_v2
.PHONY: compare_toco_inception_resnet_v2_float compare_toco_inception_resnet_v2_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor
# TOOLS_BASE := $(TFLITE_ROOT_PATH)/sandbox/mnist/tools

########################################################
# for inception_resnet_v2
########################################################
download_inception_resnet_v2:
	@ wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz -P $(QUANTOR_BASE)/inception_resnet_v2
	@ tar xvf $(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2_2016_08_30.tar.gz -C $(QUANTOR_BASE)/inception_resnet_v2
	@ mv $(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt \
		$(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2.ckpt

eval_inception_resnet_v2:
	@ PYTHONPATH=${TF_SLIM_BASE} \
	  python eval_slim_debugger.py \
		--checkpoint_path=$(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2.ckpt \
		--eval_dir=$(QUANTOR_BASE)/inception_resnet_v2 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=inception_resnet_v2 --max_num_batches=200

quantor_inception_resnet_v2: ${QUANTOR_INCPETIONRESNETV2_TARGETS}

# sub targets
freeze_inception_resnet_v2:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=inception_resnet_v2 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2_inf_graph.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inception_resnet_v2/inception_resnet_v2.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inception_resnet_v2/frozen_inception_resnet_v2.pb \
		--output_node_names=InceptionResnetV2/Logits/Predictions
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/inception_resnet_v2/frozen_inception_resnet_v2.pb

eval_inception_resnet_v2_frozen:
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionResnetV2/Logits/Predictions \
		--input_size=299 \
		--frozen_pb=$(QUANTOR_BASE)/inception_resnet_v2/frozen_inception_resnet_v2.pb --max_num_batches=200

quantor_inception_resnet_v2_frozen:
	@ python $(QUANTOR_BASE)/quantor_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_resnet_v2/frozen_inception_resnet_v2.pb \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--output_dir=$(QUANTOR_BASE)/inception_resnet_v2/quantor --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/inception_resnet_v2/summary/$@
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inception_resnet_v2/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inception_resnet_v2/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inception_resnet_v2/quantor/frozen.pb \
		--output_node_names=InceptionV3/Predictions/Reshape
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/inception_resnet_v2/quantor/frozen.pb
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--frozen_pb=$(QUANTOR_BASE)/inception_resnet_v2/quantor/frozen.pb --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/inception_resnet_v2/quantor/summary/$@

# TODO(yumaokao): should remove --allow_custom_ops after QUANTIZED is added
toco_quantor_inception_resnet_v2:
	@ mkdir -p $(QUANTOR_BASE)/inception_resnet_v2/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inception_resnet_v2/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inception_resnet_v2/quantor/model.lite \
		--mean_values=128 --std_values=127 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape --input_shapes=1,299,299,3 \
		--default_ranges_min=0 --default_ranges_max=6 --partial_quant --allow_custom_ops \
		--dump_graphviz=$(QUANTOR_BASE)/inception_resnet_v2/quantor/dots

toco_inception_resnet_v2:
	@ mkdir -p $(QUANTOR_BASE)/inception_resnet_v2/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inception_resnet_v2/frozen_inception_resnet_v2.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inception_resnet_v2/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape --input_shapes=1,299,299,3 \
		--dump_graphviz=$(QUANTOR_BASE)/inception_resnet_v2/dots

eval_quantor_inception_resnet_v2_tflite:
	@ echo $@
	@ python $(QUANTOR_BASE)/eval_tflite.py \
		--summary_dir=$(QUANTOR_BASE)/inception_resnet_v2/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inception_resnet_v2/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=299

eval_inception_resnet_v2_tflite:
	@ echo $@
	@ python $(QUANTOR_BASE)/eval_tflite.py \
		--summary_dir=$(QUANTOR_BASE)/inception_resnet_v2/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inception_resnet_v2/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=299


########################################################
# compare_toco
########################################################
compare_toco_inception_resnet_v2_float:
	@ python $(QUANTOR_BASE)/compare_toco.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_resnet_v2/frozen_inception_resnet_v2.pb \
		--max_num_batches=100 \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=299 \
		--evaluation_mode=accuracy \
		--dump_data=False

compare_toco_inception_resnet_v2_uint8:
	@ python $(QUANTOR_BASE)/compare_toco.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_resnet_v2/quantor/frozen.pb \
		--max_num_batches=100 \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=uint8 \
		--input_size=299 \
		--evaluation_mode=accuracy \
		--dump_data=False
