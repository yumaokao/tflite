DATASET_BASE:=/home/tony/tflite/datasets
TF_BASE:=/home/tony/tflite/tensorflow

# vgg16_opt.pb can be found in /proj/mtk06790/shared/models/quantor/Lmaster_vgg16.tar.gz

optimize_vgg_16:
	@ mkdir -p ./vgg16/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=./vgg16/vgg16.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TENSORFLOW_GRAPHDEF \
		--output_file=./vgg16/vgg16_opt.pb \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=input \
		--output_arrays=prob \
		--input_shapes=1,224,224,3 \
		--dump_graphviz=./vgg16/dots

eval_vgg_16:
	@ echo $@
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=prob \
		--input_size=224 --preprocess_name=vgg_caffe \
		--input_node_name=input \
		--labels_offset=1 \
		--frozen_pb=./vgg16/vgg16_opt.pb \
		--max_num_batches=1000  --batch_size=1

quantize_vgg_16:
	@ echo $@
	@ quantor_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--frozen_pb=./vgg16/vgg16_opt.pb \
		--output_node_name=prob \
		--input_node_name=input \
		--input_size=224 --preprocess_name=vgg_caffe \
		--labels_offset=1 \
		--output_dir=./vgg16/quantor \
		--max_num_batches=200 \
		--batch_size=50
	@ python $(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=./vgg16/quantor/quantor.pb \
		--input_checkpoint=./vgg16/quantor/model.ckpt \
		--input_binary=true --output_graph=./vgg16/quantor/frozen.pb \
		--output_node_names=prob
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=prob \
		--input_node_name=input \
		--input_size=224 --preprocess_name=vgg_caffe \
		--labels_offset=1 \
		--frozen_pb=./vgg16/quantor/frozen.pb --max_num_batches=1000 --batch_size=1

eval_quantize_vgg_16:
	@ eval_frozen \
		--dataset_name=imagenet \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--output_node_name=prob \
		--input_node_name=input \
		--input_size=224 --preprocess_name=vgg_caffe \
		--labels_offset=1 \
		--frozen_pb=./vgg16/quantor/frozen.pb --max_num_batches=1000 --batch_size=1

toco_vgg_16:
	@ mkdir -p ./vgg16/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=./vgg16/vgg16_opt.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=./vgg16/float_model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT \
		--input_arrays=input \
		--output_arrays=prob \
		--input_shapes=1,224,224,3 \
		--dump_graphviz=./vgg16/dots

eval_vgg_16_tflite:
	@ echo $@
	@ eval_tflite \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=./vgg16/float_model.lite \
		--inference_type=float --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=vgg_caffe \
		--labels_offset=1 \
		--max_num_batches=1000 --input_size=224

toco_quantize_vgg_16:
	@ mkdir -p ./vgg16/quantor/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=./vgg16/quantor/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF \
		--output_format=TFLITE \
		--output_file=./vgg16/quantor/model.lite \
		--inference_type=QUANTIZED_UINT8 \
		--inference_input_type=QUANTIZED_UINT8 \
		--mean_values=114.8 --std_values=1.0 \
		--input_arrays=input \
		--output_arrays=prob \
		--input_shapes=1,224,224,3 \
		--dump_graphviz=./vgg16/quantor/dots

eval_quantize_vgg_16_tflite:
	@ echo $@
	@ eval_tflite \
		--dataset_name=imagenet --dataset_split_name=test \
		--dataset_dir=$(DATASET_BASE)/imagenet \
		--tflite_model=./vgg16/quantor/model.lite \
		--inference_type=uint8 --tensorflow_dir=$(TF_BASE) \
		--preprocess_name=vgg_caffe \
		--labels_offset=1 \
		--max_num_batches=1000 --input_size=224

