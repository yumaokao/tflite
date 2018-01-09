.PHONY: all build clean images

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
MODELS_BASE := $(TFLITE_ROOT_PATH)/sandbox/models
TOOLS_BASE := $(TFLITE_ROOT_PATH)/sandbox/mnist/tools

VARIABLE_TARGETS := mobilenet_1.0

.PHONY: $(VARIABLE_TARGETS)

## functions ##
define freeze_graph
	@ echo "freeze_graph"
	$(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$1/model.pb \
		--input_checkpoint=$1/ckpts/model.ckpt \
		--input_binary=true \
		--output_graph=$1/frozen.pb \
		--output_node_names=$2
endef

define toco
	@ echo "toco"
	@ mkdir -p $1/dots
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/$2 \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=10,224,224,3 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef

define compare_tflite
	@ echo "compare_tflite"
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/compare_tflite \
		--tflite_file=$1/export/model.lite \
		--batch_xs=$1/export/batch_xs.npy \
		--batch_ys=$1/export/ys.npy $2
endef


all:
	@ echo "all models"

clean:
	@ rm -rf */summary */dots */export */ckpts */*.pb

build:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite

images:
	@ mkdir -p $@
	@ wget "http://blogs.vault.com/media/3782983/sleeping_cat.jpg" -O $@/cat_0.jpg
	@ wget "http://itchmo.com/wp-content/uploads/2007/06/cat-facts.jpg" -O $@/cat_1.jpg
	@ wget "https://pbs.twimg.com/media/CEp7EgeWIAM-L8Q.jpg" -O $@/cat_2.jpg
	@ wget "https://404store.com/2017/05/06/worlds-most-beautiful-cats-52-57fc9cb29d61b__700.jpg" -O $@/cat_3.jpg
	@ wget "https://i.imgur.com/IGtpDrt.jpg" -O $@/cat_4.jpg
	@ wget "https://cdn.theatlantic.com/assets/media/img/mt/2017/04/GettyImages_138468381/lead_960.jpg" -O $@/cat_5.jpg
	@ wget "http://216.243.143.215/images/uploads/calico%20kitten.jpg" -O $@/cat_6.jpg
	@ wget "http://www.lifewithcats.tv/wp-content/uploads/2011/08/storm.jpg" -O $@/cat_7.jpg
	@ wget "https://pbs.twimg.com/media/BdpQBgkIEAAt_X-.jpg" -O $@/cat_8.jpg
	@ wget "http://www.lifewithcats.tv/wp-content/uploads/2013/03/528280_549595008418823_1073077699_n.jpg" -O $@/cat_9.jpg


# need to download backbone.tgz and extracted
mobilenet_1.0:
	@ mkdir -p $@/export
	@ cp backbone/mobilenet_1.0/mobilenet_v1_1.0_224_frozen.pb $@/frozen.pb
	@ python $(TOOLS_BASE)/drop_dropouts.py $@/frozen.pb
#	MobilenetV1/MobilenetV1/Conv2d_0/convolution 		OK
#	MobilenetV1/MobilenetV1/Conv2d_0/Relu6			OK
#	MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise	OK
#	MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6	OK
#	MobilenetV1/MobilenetV1/Conv2d_1_pointwise/convolution	Failed (batch=10) (ANN OK)
#	MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6	Falied
#	...
# 	MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd 		Failed
	@ python $@/$@.py $@/frozen-nodropout.pb --y MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd
	$(call toco,$(MODELS_BASE)/$@,frozen-nodropout.pb,input,MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd,)
	$(call compare_tflite,$(MODELS_BASE)/$@,--ignore 1)
