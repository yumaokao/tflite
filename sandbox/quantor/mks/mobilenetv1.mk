.PHONY: download_mobilenetV1_224 eval_mobilenetV1_224
.PHONY: freeze_mobilenetV1_224 eval_mobilenetV1_224_frozen
.PHONY: quantor_mobilenetV1_224_frozen toco_mobilenetV1_224
.PHONY: eval_mobilenetV1_224_tflite
.PHONY: compare_toco_mobilenetV1_224_float compare_toco_mobilenetV1_224_uint8

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
# for mobilenetV1_224
########################################################
download_mobilenetV1_224:
	@ wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz -P $(QUANTOR_BASE)/mobilenetV1_224
	@ tar xvf $(QUANTOR_BASE)/mobilenetV1_224/mobilenet_v1_1.0_224_2017_06_14.tar.gz -C $(QUANTOR_BASE)/mobilenetV1_224

eval_mobilenetV1_224:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/mobilenetV1_224/mobilenet_v1_1.0_224.ckpt \
		--eval_dir=$(QUANTOR_BASE)/mobilenetV1_224 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=mobilenet_v1 --max_num_batches=50

freeze_mobilenetV1_224:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=mobilenet_v1 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/mobilenetV1_224/mobilenetV1_224_inf_graph.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/mobilenetV1_224/mobilenetV1_224_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/mobilenetV1_224/mobilenetV1_224_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/mobilenetV1_224/mobilenet_v1_1.0_224.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb

eval_mobilenetV1_224_frozen:
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb --max_num_batches=50 \
		# --summary_dir=$(QUANTOR_BASE)/mobilenetV1_224/summary/$@

quantor_mobilenetV1_224_frozen:
	@ python $(QUANTOR_BASE)/quantor_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--output_dir=$(QUANTOR_BASE)/mobilenetV1_224/quantor --max_num_batches=50
		# --summary_dir=$(QUANTOR_BASE)/mobilenetV1_224/summary/$@
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/mobilenetV1_224/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/mobilenetV1_224/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/mobilenetV1_224/quantor/frozen.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/mobilenetV1_224/quantor/frozen.pb
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/quantor/frozen.pb --max_num_batches=50
		# --summary_dir=$(QUANTOR_BASE)/mobilenetV1_224/quantor/summary/$@

toco_mobilenetV1_224:
	@ mkdir -p $(QUANTOR_BASE)/mobilenetV1_224/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/mobilenetV1_224/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=MobilenetV1/Predictions/Reshape --input_shapes=1,224,224,3 \
		--dump_graphviz=$(QUANTOR_BASE)/mobilenetV1_224/dots

eval_mobilenetV1_224_tflite:
	@ python $(QUANTOR_BASE)/eval_tflite.py \
		--summary_dir=$(QUANTOR_BASE)/mobilenetV1_224/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/mobilenetV1_224/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=224

########################################################
# compare_toco
########################################################
compare_toco_mobilenetV1_224_float:
	@ python $(QUANTOR_BASE)/compare_toco.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb \
		--max_num_batches=100 \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=224 \
		--evaluation_mode=accuracy \
		--dump_data=False

compare_toco_mobilenetV1_224_uint8:
	@ python $(QUANTOR_BASE)/compare_toco.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/quantor/frozen.pb \
		--max_num_batches=100 \
		--output_node_name=MobilenetV1/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=uint8 \
		--input_size=224 \
		--evaluation_mode=accuracy \
		--dump_data=False
