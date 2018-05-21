# The dispflownet implementation is derived from https://github.com/fedor-chervinskii/dispflownet-tf commit ed2cb23f51022b420e75455a4d1137987702c5c0,
# and we make the following modification on dispnet.py 
#
# 1. Disable the usage of shift_corr_module
# 2. Fix the network structure if is_corr is False
#
#      with tf.variable_scope("up2"):
#      -        concat2, predict3 = upsampling_block(concat3, conv2a, 128, 64, 128)
#      +        if is_corr:
#      +            concat2, predict3 = upsampling_block(concat3, conv2a, 128, 64, 128)
#      +        else:
#      +            concat2, predict3 = upsampling_block(concat3, conv2, 128, 64, 128)
#      with tf.variable_scope("up1"):
#      -        concat1, predict2 = upsampling_block(concat2, conv1a, 64, 32, 64)
#      +        if is_corr:
#      +            concat1, predict2 = upsampling_block(concat2, conv1a, 64, 32, 64)
#      +        else:
#      +            concat1, predict2 = upsampling_block(concat2, conv1, 64, 32, 64)
#
# 3. Add extra argument to decide whether to use correlation
# 4. Do not shuffle the test data in test mode
#

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow
DISPFLOW_BASE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

download_dataset:
	@ mkdir -p $(DISPFLOW_BASE)/datasets/flyingthings3d
	@ wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar -P $(DISPFLOW_BASE)/datasets/flyingthings3d
	@ wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2 -P $(DISPFLOW_BASE)/datasets/flyingthings3d
	@ cd $(DISPFLOW_BASE)/datasets/flyingthings3d && tar xvf flyingthings3d__frames_cleanpass.tar
	@ cd $(DISPFLOW_BASE)/datasets/flyingthings3d && tar xvf flyingthings3d__disparity.tar.bz2

train_dispflownet:
	@ python train.py \
		--data_path=$(DISPFLOW_BASE)/datasets/flyingthings3d/ \
		--ckpt=$(DISPFLOW_BASE)/model/ckpts \
		--batch_size=4 \
		--log_step=100 \
		--save_step=10000 \
		--n_steps=1000000 \
		--weight_schedule

test_dispflownet:
	@ python test_flyingthings.py \
		--data_path=$(DISPFLOW_BASE)/datasets/flyingthings3d/ \
		--ckpt=$(DISPFLOW_BASE)/model/ckpts \
		--batch_size=1 \
		--log_step=100

# 2.197989 [1000000-without-schedule]
# 2.114056 [1000000-with-schedule]
export_dispflownet:
	@ python export.py \
		--ckpt=$(DISPFLOW_BASE)/model/ckpts \
		--output=$(DISPFLOW_BASE)/model
	@ python eval_frozen.py \
		--data_path=$(DISPFLOW_BASE)/datasets/flyingthings3d \
		--frozen_model=$(DISPFLOW_BASE)/model/frozen.pb \
		--input_nodes=left_img:right_img \
		--batch_size=1 \
		--log_step=100 \
		--output_node=prediction/conv/BiasAdd

# 6.450870 [1000000-with-schedule] @ 10000-direct_quant
direct_quantize_dispflownet:
	@ mkdir -p $(DISPFLOW_BASE)/model/quantor
	@ python direct_quantize_frozen.py \
		--data_path=$(DISPFLOW_BASE)/datasets/flyingthings3d \
		--frozen_model=$(DISPFLOW_BASE)/model/frozen.pb \
		--input_nodes=left_img:right_img \
		--batch_size=1 \
		--num_batches=10000 \
		--log_step=100 \
		--output_dir=$(DISPFLOW_BASE)/model/quantor \
		--output_node=prediction/conv/act_quant/FakeQuantWithMinMaxVars
	@ save_summaries $(DISPFLOW_BASE)/model/quantor/frozen.pb
	@ python eval_frozen.py \
		--data_path=$(DISPFLOW_BASE)/datasets/flyingthings3d \
		--frozen_model=$(DISPFLOW_BASE)/model/quantor/frozen.pb \
		--input_nodes=left_img:right_img \
		--batch_size=1 \
		--log_step=100 \
		--output_node=prediction/conv/act_quant/FakeQuantWithMinMaxVars
