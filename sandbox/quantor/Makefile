.PHONY: all clean build
.PHONY: dataset_imagenet dataset_cifar10 dataset_flowers dataset_mnist
.PHONY: train_cifarnet eval_cifarnet freeze_graph eval_cifarnet_frozen
.PHONY: toco_quantor_cifarnet toco_cifarnet eval_cifarnet_tflite eval_quantor_cifarnet_tflite

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor
TOOLS_BASE := $(TFLITE_ROOT_PATH)/sandbox/mnist/tools

all:
	@ echo "all models"

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:summarize_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite

clean:
	@ rm -rf $(QUANTOR_BASE)/cifarnet


# datasets
dataset_imagenet:
	@ cd $(TF_SLIM_BASE) && bazel build download_and_convert_imagenet
	@ DATA_DIR=$(DATASET_BASE)/imagenet \
		&& cd $(TF_SLIM_BASE) \
		&& $(TF_SLIM_BASE)/bazel-bin/download_and_convert_imagenet "$(DATASET_BASE)/imagenet"

dataset_cifar10:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=cifar10 --dataset_dir="$(DATASET_BASE)/cifar10"

dataset_flowers:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=flowers --dataset_dir="$(DATASET_BASE)/flowers"

dataset_mnist:
	@ python $(TF_SLIM_BASE)/download_and_convert_data.py \
		--dataset_name=mnist --dataset_dir="$(DATASET_BASE)/mnist"

########################################################
# slim cifarnet
#   add --clone_on_cpu to run training on cpu
########################################################
train_cifarnet:
	@ cd $(TF_SLIM_BASE) && python train_image_classifier.py \
		--train_dir=$(QUANTOR_BASE)/cifarnet --dataset_name=cifar10 \
		--dataset_split_name=train --dataset_dir=$(DATASET_BASE)/cifar10 \
		--model_name=cifarnet --preprocessing_name=cifarnet \
		--max_number_of_steps=100000 --batch_size=128 \
		--save_interval_secs=120 --save_summaries_secs=120 \
		--log_every_n_steps=100 --optimizer=sgd \
		--learning_rate=0.1 --learning_rate_decay_factor=0.1 \
		--num_epochs_per_decay=200 --weight_decay=0.004

eval_cifarnet:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/cifarnet --eval_dir=$(QUANTOR_BASE)/cifarnet \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 --model_name=cifarnet

freeze_cifarnet:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=cifarnet --dataset_name=cifar10 \
		--output_file=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/cifarnet/cifarnet_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/cifarnet/model.ckpt-100000 \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb \
		--output_node_names=CifarNet/Predictions/Reshape
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb

# eval frozen
eval_cifarnet_frozen:
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--frozen_pb=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb

quantor_cifarnet_frozen:
	@ python $(QUANTOR_BASE)/quantor_frozen.py \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--frozen_pb=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb \
		--output_dir=$(QUANTOR_BASE)/cifarnet/quantor
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/cifarnet/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/cifarnet/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/cifarnet/quantor/frozen.pb \
		--output_node_names=CifarNet/Predictions/Reshape
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/cifarnet/quantor/frozen.pb
	@ python $(QUANTOR_BASE)/eval_frozen.py \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/quantor/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--frozen_pb=$(QUANTOR_BASE)/cifarnet/quantor/frozen.pb

# TODO(yumaokao): should remove --allow_custom_ops after QUANTIZED is added
#   CifarNet/conv1/Relu
#   CifarNet/Predictions/Reshape
toco_quantor_cifarnet:
	@ mkdir -p $(QUANTOR_BASE)/cifarnet/quantor/dots
	$(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/cifarnet/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/cifarnet/quantor/model.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--mean_values=128 --std_values=127 \
		--output_arrays=CifarNet/Predictions/Reshape --input_shapes=1,32,32,3 \
		--default_ranges_min=0 --default_ranges_max=6 --partial_quant --allow_custom_ops \
		--dump_graphviz=$(QUANTOR_BASE)/cifarnet/quantor/dots

toco_cifarnet:
	@ mkdir -p $(QUANTOR_BASE)/cifarnet/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/cifarnet/frozen_cifarnet.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/cifarnet/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=CifarNet/Predictions/Reshape --input_shapes=1,32,32,3 \
		--dump_graphviz=$(QUANTOR_BASE)/cifarnet/dots

eval_quantor_cifarnet_tflite:
	@ python $(QUANTOR_BASE)/eval_tflite.py \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/quantor/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--tflite_model=$(QUANTOR_BASE)/cifarnet/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE)

eval_cifarnet_tflite:
	@ python $(QUANTOR_BASE)/eval_tflite.py \
		--summary_dir=$(QUANTOR_BASE)/cifarnet/quantor/summary/$@ \
		--dataset_name=cifar10 --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/cifar10 \
		--tflite_model=$(QUANTOR_BASE)/cifarnet/float_model.lite --tensorflow_dir=$(TF_BASE)


########################################################
# for mobilenetV1_224
########################################################
download_mobilenetV1_224:
	@ wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz -P $(QUANTOR_BASE)/mobilenetV1_224	
	@ tar xvf $(QUANTOR_BASE)/mobilenetV1_224/mobilenet_v1_1.0_224_2017_06_14.tar.gz -C $(QUANTOR_BASE)/mobilenetV1_224

#71.2%
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

#69.2%
eval_mobilenetV1_224_frozen:
	@ python $(QUANTOR_BASE)/eval_frozen_imagenet.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_names=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb --max_num_batches=50 \
		# --summary_dir=$(QUANTOR_BASE)/mobilenetV1_224/summary/$@

#5.04%
quantor_mobilenetV1_224_frozen:
	@ python $(QUANTOR_BASE)/quantor_frozen_imagenet.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/frozen_mobilenetV1_224.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--output_dir=$(QUANTOR_BASE)/mobilenetV1_224/quantor --max_num_batches=50
		# --summary_dir=$(QUANTOR_BASE)/mobilenetV1_224/summary/$@
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/mobilenetV1_224/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/mobilenetV1_224/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/mobilenetV1_224/quantor/frozen.pb \
		--output_node_names=MobilenetV1/Predictions/Reshape
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/mobilenetV1_224/quantor/frozen.pb
	@ python $(QUANTOR_BASE)/eval_frozen_imagenet.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_names=MobilenetV1/Predictions/Reshape \
		--input_size=224 \
		--frozen_pb=$(QUANTOR_BASE)/mobilenetV1_224/quantor/frozen.pb --max_num_batches=50
		# --summary_dir=$(QUANTOR_BASE)/mobilenetV1_224/quantor/summary/$@


########################################################
# for inceptionV3
########################################################
download_inceptionV3:
	@ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz -P $(QUANTOR_BASE)/inceptionV3
	@ tar xvf $(QUANTOR_BASE)/inceptionV3/inception_v3_2016_08_28.tar.gz -C $(QUANTOR_BASE)/inceptionV3

eval_inceptionV3:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/inceptionV3/inception_v3.ckpt \
		--eval_dir=$(QUANTOR_BASE)/inceptionV3 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=inception_v3 --max_num_batches=50

freeze_inceptionV3:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=inception_v3 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/inceptionV3/inceptionV3_inf_graph.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/inceptionV3/inceptionV3_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inceptionV3/inceptionV3_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inceptionV3/inception_v3.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inceptionV3/frozen_inceptionV3.pb \
		--output_node_names=InceptionV3/Predictions/Reshape
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/inceptionV3/frozen_inceptionV3.pb
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/inceptionV3/frozen_inceptionV3.pb

eval_inceptionV3_frozen:
	@ python $(QUANTOR_BASE)/eval_frozen_imagenet.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_names=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--frozen_pb=$(QUANTOR_BASE)/inceptionV3/frozen_inceptionV3.pb --max_num_batches=50
		# --summary_dir=$(QUANTOR_BASE)/inceptionV3/summary/$@

quantor_inceptionV3_frozen:
	@ python $(QUANTOR_BASE)/quantor_frozen_imagenet.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inceptionV3/frozen_inceptionV3.pb \
		--output_node_names=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--output_dir=$(QUANTOR_BASE)/inceptionV3/quantor --max_num_batches=50
		# --summary_dir=$(QUANTOR_BASE)/inceptionV3/summary/$@
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inceptionV3/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inceptionV3/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inceptionV3/quantor/frozen.pb \
		--output_node_names=InceptionV3/Predictions/Reshape
	@ python $(TOOLS_BASE)/save_summaries.py $(QUANTOR_BASE)/inceptionV3/quantor/frozen.pb
	@ python $(QUANTOR_BASE)/eval_frozen_imagenet.py \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_names=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--frozen_pb=$(QUANTOR_BASE)/inceptionV3/quantor/frozen.pb --max_num_batches=50
		# --summary_dir=$(QUANTOR_BASE)/inceptionV3/quantor/summary/$@

# TODO(yumaokao): should remove --allow_custom_ops after QUANTIZED is added
toco_quantor_inceptionV3:
	@ mkdir -p $(QUANTOR_BASE)/inceptionV3/quantor/dots
	$(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inceptionV3/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inceptionV3/quantor/model.lite \
		--mean_values=128 --std_values=127 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape --input_shapes=1,299,299,3 \
		--default_ranges_min=0 --default_ranges_max=6 --partial_quant --allow_custom_ops \
		--dump_graphviz=$(QUANTOR_BASE)/inceptionV3/quantor/dots

toco_inceptionV3:
	@ mkdir -p $(QUANTOR_BASE)/inceptionV3/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inceptionV3/frozen_inceptionV3.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inceptionV3/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape --input_shapes=1,299,299,3 \
		--dump_graphviz=$(QUANTOR_BASE)/inceptionV3/dots

eval_quantor_inceptionV3_tflite:
	@ python $(QUANTOR_BASE)/eval_tflite_imagenet.py \
		--summary_dir=$(QUANTOR_BASE)/inceptionV3/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inceptionV3/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=50 --input_size=299

eval_inceptionV3_tflite:
	@ python $(QUANTOR_BASE)/eval_tflite_imagenet.py \
		--summary_dir=$(QUANTOR_BASE)/inceptionV3/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inceptionV3/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=50 --input_size=299
