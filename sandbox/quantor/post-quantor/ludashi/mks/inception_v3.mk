DATASET_BASE:=/home/tony/tflite/datasets
TF_BASE:=/home/tony/tflite/tensorflow

# inception_v3_2016_08_28_frozen.pb can be found in /proj/mtk06790/shared/models/quantor/Lmaster_inception_v3.tar.gz

optimize_inception_v3:
	@ mkdir -p ./inception_v3/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=./inception_v3/inception_v3_2016_08_28_frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TENSORFLOW_GRAPHDEF \
		--output_file=./inception_v3/inception_v3_opt.pb \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape_1 \
		--input_shapes=1,299,299,3 \
		--dump_graphviz=./inception_v3/dots

eval_inception_v3:
	@ echo $@
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV3/Predictions/Reshape_1 \
		--input_size=299 --preprocess_name=inception \
		--input_node_name=input \
		--frozen_pb=./inception_v3/inception_v3_opt.pb \
		--max_num_batches=1000  --batch_size=1

quantize_inception_v3:
	@ echo $@
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=./inception_v3/inception_v3_opt.pb \
		--output_node_name=InceptionV3/Predictions/Reshape_1 \
		--input_node_name=input \
		--input_size=299 --preprocess_name=inception \
		--output_dir=./inception_v3/quantor \
		--max_num_batches=200 \
		--batch_size=50
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=./inception_v3/quantor/quantor.pb \
		--input_checkpoint=./inception_v3/quantor/model.ckpt \
		--input_binary=true --output_graph=./inception_v3/quantor/frozen.pb \
		--output_node_names=InceptionV3/Predictions/Reshape_1
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV3/Predictions/Reshape_1 \
		--input_node_name=input \
		--input_size=299 --preprocess_name=inception \
		--frozen_pb=./inception_v3/quantor/frozen.pb --max_num_batches=1000 --batch_size=1

eval_quantize_inception_v3:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=InceptionV3/Predictions/Reshape_1 \
		--input_node_name=input \
		--input_size=299 --preprocess_name=inception \
		--frozen_pb=./inception_v3/quantor/frozen.pb --max_num_batches=1000 --batch_size=1

toco_inception_v3:
	@ mkdir -p ./inception_v3/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=./inception_v3/inception_v3_opt.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=./inception_v3/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape_1 \
		--input_shapes=1,299,299,3 \
		--dump_graphviz=./inception_v3/dots

eval_inception_v3_tflite:
	@ echo $@
	@ eval_tflite \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=./inception_v3/float_model.lite \
		--inference_type=float --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=inception \
		--max_num_batches=1000 --input_size=299

toco_quantize_inception_v3:
	@ mkdir -p ./inception_v3/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=./inception_v3/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=./inception_v3/quantor/model.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 \
		--mean_values=128 --std_values=127 \
		--input_arrays=input \
		--output_arrays=InceptionV3/Predictions/Reshape_1 \
		--input_shapes=1,299,299,3 \
		--dump_graphviz=./inception_v3/quantor/dots

eval_quantize_inception_v3_tflite:
	@ echo $@
	@ eval_tflite \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=./inception_v3/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=inception \
		--max_num_batches=1000 --input_size=299

