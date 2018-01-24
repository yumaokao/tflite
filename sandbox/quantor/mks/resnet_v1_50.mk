# float model
QUANTOR_RESNET_V1_50_TARGETS := freeze_resnet_v1_50
QUANTOR_RESNET_V1_50_TARGETS += eval_resnet_v1_50_frozen
# float (fake quanted) model
QUANTOR_RESNET_V1_50_TARGETS += toco_resnet_v1_50
QUANTOR_RESNET_V1_50_TARGETS += eval_resnet_v1_50_tflite
# uint8 model
QUANTOR_RESNET_V1_50_TARGETS += quantor_resnet_v1_50_frozen
QUANTOR_RESNET_V1_50_TARGETS += toco_quantor_resnet_v1_50
QUANTOR_RESNET_V1_50_TARGETS += eval_quantor_resnet_v1_50_tflite

.PHONY: download_resnet_v1_50 eval_resnet_v1_50
.PHONY: ${QUANTOR_RESNET_V1_50_TARGETS}
.PHONY: quantor_resnet_v1_50
.PHONY: compare_toco_resnet_v1_50_float compare_toco_resnet_v1_50_uint8

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
# for resnet_v1_50
########################################################
download_resnet_v1_50:
	@ wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz -P $(QUANTOR_BASE)/resnet_v1_50
	@ tar xvf $(QUANTOR_BASE)/resnet_v1_50/resnet_v1_50_2016_08_28.tar.gz -C $(QUANTOR_BASE)/resnet_v1_50

eval_resnet_v1_50:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/resnet_v1_50/resnet_v1_50.ckpt \
		--eval_dir=$(QUANTOR_BASE)/resnet_v1_50 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--labels_offset=1 \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=resnet_v1_50 --max_num_batches=200

quantor_resnet_v1_50: ${QUANTOR_RESNET_V1_50_TARGETS}

# sub targets
freeze_resnet_v1_50:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr --labels_offset=1 \
		--model_name=resnet_v1_50 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/resnet_v1_50/resnet_v1_50_inf_graph.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/resnet_v1_50/resnet_v1_50_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/resnet_v1_50/resnet_v1_50_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/resnet_v1_50/resnet_v1_50.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb \
		--output_node_names=resnet_v1_50/logits/BiasAdd
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
		--in_graph=$(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb \
		--out_graph=$(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50_tmp.pb \
		--inputs=input \
		--outputs=resnet_v1_50/logits/BiasAdd \
		--transforms=fold_old_batch_norms
	@ mv $(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50_tmp.pb $(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb

eval_resnet_v1_50_frozen:
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=resnet_v1_50/logits/BiasAdd \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb --max_num_batches=200

quantor_resnet_v1_50_frozen:
	@ python $(QUANTOR_BASE)/quantor_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb \
		--output_node_name=resnet_v1_50/logits/BiasAdd \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--output_dir=$(QUANTOR_BASE)/resnet_v1_50/quantor --max_num_batches=200
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/resnet_v1_50/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/resnet_v1_50/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/resnet_v1_50/quantor/frozen.pb \
		--output_node_names=resnet_v1_50/logits/BiasAdd
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/resnet_v1_50/quantor/frozen.pb
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=resnet_v1_50/logits/BiasAdd \
		--input_size=224 --labels_offset=1 --preprocess_name=vgg \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_50/quantor/frozen.pb --max_num_batches=200

toco_quantor_resnet_v1_50:
	@ mkdir -p $(QUANTOR_BASE)/resnet_v1_50/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v1_50/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet_v1_50/quantor/model.lite \
		--mean_values=128 --std_values=127 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=InceptionResnetV2/Logits/Predictions --input_shapes=1,299,299,3 \
		--default_ranges_min=0 --default_ranges_max=10 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet_v1_50/quantor/dots

toco_resnet_v1_50:
	@ mkdir -p $(QUANTOR_BASE)/resnet_v1_50/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/resnet_v1_50/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=InceptionResnetV2/Logits/Predictions --input_shapes=1,299,299,3 \
		--dump_graphviz=$(QUANTOR_BASE)/resnet_v1_50/dots

eval_quantor_resnet_v1_50_tflite:
	@ echo $@
	@ python $(QUANTOR_BASE)/eval_tflite.py \
		--summary_dir=$(QUANTOR_BASE)/resnet_v1_50/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet_v1_50/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=299

eval_resnet_v1_50_tflite:
	@ echo $@
	@ python $(QUANTOR_BASE)/eval_tflite.py \
		--summary_dir=$(QUANTOR_BASE)/resnet_v1_50/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/resnet_v1_50/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=299


########################################################
# compare_toco
########################################################
compare_toco_resnet_v1_50_float:
	@ python $(QUANTOR_BASE)/compare_toco.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_50/frozen_resnet_v1_50.pb \
		--max_num_batches=100 \
		--output_node_name=resnet_v1_50/logits/BiasAdd \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=224 \
		--evaluation_mode=accuracy \
		--dump_data=False

compare_toco_resnet_v1_50_uint8:
	@ python $(QUANTOR_BASE)/compare_toco.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/resnet_v1_50/quantor/frozen.pb \
		--max_num_batches=100 \
		--output_node_name=resnet_v1_50/logits/BiasAdd \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=uint8 \
		--input_size=224 \
		--evaluation_mode=accuracy \
		--dump_data=False
