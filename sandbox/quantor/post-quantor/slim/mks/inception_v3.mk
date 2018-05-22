# float model
QUANTOR_INCEPTIONV3_TARGETS := freeze_inception_v3
QUANTOR_INCEPTIONV3_TARGETS += eval_inception_v3_frozen
# float model
QUANTOR_INCEPTIONV3_TARGETS += toco_inception_v3
QUANTOR_INCEPTIONV3_TARGETS += eval_inception_v3_tflite
# uint8 model
QUANTOR_INCEPTIONV3_TARGETS += quantor_inception_v3_frozen
QUANTOR_INCEPTIONV3_TARGETS += toco_quantor_inception_v3
QUANTOR_INCEPTIONV3_TARGETS += eval_quantor_inception_v3_tflite

.PHONY: download_inception_v3 eval_inception_v3
.PHONY: ${QUANTOR_INCEPTIONV3_TARGETS}
.PHONY: quantor_inception_v3
.PHONY: compare_toco_inception_v3_float compare_toco_inception_v3_uint8

########################################################
# should already defined these variables
########################################################
# TFLITE_ROOT_PATH := /home/tflite
# TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
# TF_SLIM_BASE := $(TFLITE_ROOT_PATH)/models/research/slim
# DATASET_BASE := $(TFLITE_ROOT_PATH)/datasets
# QUANTOR_BASE := $(TFLITE_ROOT_PATH)/sandbox/quantor

########################################################
# for inception_v3
########################################################
download_inception_v3:
	@ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz -P $(QUANTOR_BASE)/inception_v3
	@ tar xvf $(QUANTOR_BASE)/inception_v3/inception_v3_2016_08_28.tar.gz -C $(QUANTOR_BASE)/inception_v3

eval_inception_v3:
	@ cd $(TF_SLIM_BASE) && python eval_image_classifier.py \
		--checkpoint_path=$(QUANTOR_BASE)/inception_v3/inception_v3.ckpt \
		--eval_dir=$(QUANTOR_BASE)/inception_v3 \
		--dataset_name=imagenet --dataset_split_name=validation \
		--dataset_dir=$(DATASET_BASE)/imagenet --model_name=inception_v3 --max_num_batches=200

quantor_inception_v3: ${QUANTOR_INCEPTIONV3_TARGETS}

# sub targets
freeze_inception_v3:
	@ cd $(TF_SLIM_BASE) && python export_inference_graph.py \
		--alsologtostderr \
		--model_name=inception_v3 --dataset_name=imagenet \
		--output_file=$(QUANTOR_BASE)/inception_v3/inception_v3_inf_graph.pb
	@ save_summaries $(QUANTOR_BASE)/inception_v3/inception_v3_inf_graph.pb
	@ cd $(TF_BASE) && bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inception_v3/inception_v3_inf_graph.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inception_v3/inception_v3.ckpt \
		--checkpoint_version=1 \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb \
		--output_node_names=InceptionV3/Predictions/Reshape
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
		--in_graph=$(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb
	@ save_summaries $(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb

eval_inception_v3_frozen:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--frozen_pb=$(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb --max_num_batches=200
		# --summary_dir=$(QUANTOR_BASE)/inception_v3/summary/$@

eval_inception_v3_frozen_jpeg:
	@ echo $@
	@ python process_frozen_jpeg.py \
		--frozen_pb=$(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb \
		--dataset_dir=/mnt/8TB/chialin/datasets/imagenet_jpeg \
		--batch_size=50 \
		--log_step=10 \
		--num_batches=200 \
		--labels_offset=0 \
		--preprocess_name=inception \
		--input_node_name=input \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299
	@ python eval_image_classification.py \
		--data_dir=$(QUANTOR_BASE)/inception_v3/dump_frozen_jpeg \
		--num_batches=200

quantor_inception_v3_frozen:
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--output_dir=$(QUANTOR_BASE)/inception_v3/quantor --max_num_batches=200
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$(QUANTOR_BASE)/inception_v3/quantor/quantor.pb \
		--input_checkpoint=$(QUANTOR_BASE)/inception_v3/quantor/model.ckpt \
		--input_binary=true --output_graph=$(QUANTOR_BASE)/inception_v3/quantor/frozen.pb \
		--output_node_names=InceptionV3/Predictions/Reshape
	@ save_summaries $(QUANTOR_BASE)/inception_v3/quantor/frozen.pb
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299 \
		--frozen_pb=$(QUANTOR_BASE)/inception_v3/quantor/frozen.pb --max_num_batches=200

eval_quantor_inception_v3_frozen_jpeg:
	@ echo $@
	@ python process_frozen_jpeg.py \
		--frozen_pb=$(QUANTOR_BASE)/inception_v3/quantor/frozen.pb \
		--dataset_dir=/mnt/8TB/chialin/datasets/imagenet_jpeg \
		--batch_size=50 \
		--log_step=10 \
		--num_batches=200 \
		--labels_offset=0 \
		--preprocess_name=inception \
		--input_node_name=input \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299
	@ python eval_image_classification.py \
		--data_dir=$(QUANTOR_BASE)/inception_v3/quantor/dump_frozen_jpeg \
		--num_batches=200

toco_quantor_inception_v3:
	@ mkdir -p $(QUANTOR_BASE)/inception_v3/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inception_v3/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inception_v3/quantor/model.lite \
		--mean_values=128 --std_values=127 \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape --input_shapes=10,299,299,3 \
		--dump_graphviz=$(QUANTOR_BASE)/inception_v3/quantor/dots

toco_inception_v3:
	@ mkdir -p $(QUANTOR_BASE)/inception_v3/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$(QUANTOR_BASE)/inception_v3/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape --input_shapes=1,299,299,3 \
		--dump_graphviz=$(QUANTOR_BASE)/inception_v3/dots

eval_quantor_inception_v3_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/inception_v3/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inception_v3/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=1000 --input_size=299 --batch_size=10

eval_inception_v3_tflite:
	@ echo $@
	@ eval_tflite \
		--summary_dir=$(QUANTOR_BASE)/inception_v3/quantor/summary/$@ \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=$(QUANTOR_BASE)/inception_v3/float_model.lite --tensorflow_dir=$(TF_BASE) \
		--max_num_batches=10000 --input_size=299

eval_inception_v3_tflite_jpeg:
	@ echo $@
	@ python process_tflite_jpeg.py \
		--tflite_model=$(QUANTOR_BASE)/inception_v3/float_model.lite \
		--dataset_dir=/mnt/8TB/chialin/datasets/imagenet_jpeg \
		--batch_size=1 \
		--log_step=1000 \
		--num_batches=10000 \
		--labels_offset=0 \
		--preprocess_name=inception \
		--input_node_name=input \
		--inference_type=float \
		--tensorflow_dir=$(TF_BASE) \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299
	@ python eval_image_classification.py \
		--data_dir=$(QUANTOR_BASE)/inception_v3/dump_tflite_jpeg \
		--num_batches=10000

eval_quantor_inception_v3_tflite_jpeg:
	@ echo $@
	@ python process_tflite_jpeg.py \
		--tflite_model=$(QUANTOR_BASE)/inception_v3/quantor/model.lite \
		--dataset_dir=/mnt/8TB/chialin/datasets/imagenet_jpeg \
		--batch_size=1 \
		--log_step=1000 \
		--num_batches=10000 \
		--labels_offset=0 \
		--preprocess_name=inception \
		--input_node_name=input \
		--inference_type=uint8 \
		--tensorflow_dir=$(TF_BASE) \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--input_size=299
	@ python eval_image_classification.py \
		--data_dir=$(QUANTOR_BASE)/inception_v3/quantor/dump_tflite_jpeg \
		--num_batches=10000

########################################################
# compare_toco
########################################################
compare_toco_inception_v3_float:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_v3/frozen_inception_v3.pb \
		--max_num_batches=100 \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=float \
		--input_size=299 \
		--dump_data=False

compare_toco_inception_v3_uint8:
	@ compare_toco \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=$(QUANTOR_BASE)/inception_v3/quantor/frozen.pb \
		--max_num_batches=100 \
		--output_node_name=InceptionV3/Predictions/Reshape \
		--tensorflow_dir=$(TF_BASE) \
		--toco_inference_type=uint8 \
		--input_size=299 \
		--dump_data=False
