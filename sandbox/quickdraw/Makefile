.PHONY: all clean
.PHONY: datasets

TF_BASE := /home/tflite/tensorflow
QUICKDRAW_BASE := /home/tflite/sandbox/quickdraw

QUICKDRAW_TARGETS := tutorial-lstm-1k tutorial-cudnn_lstm-10k
.PHONY: $(QUICKDRAW_TARGETS)


# functions
define train_model
	@ # echo MODEL_DIR [$1]
	@ # echo CELL_TYPE $2
	@ # echo STEPS $3
	@ mkdir -p $1
	@ python tutorial/train_model.py \
	  --training_data=datasets/tutorial_v1/training.tfrecord-?????-of-????? \
	  --eval_data=datasets/tutorial_v1/eval.tfrecord-?????-of-????? \
	  --classes_file=datasets/tutorial_v1/training.tfrecord.classes \
	  --model_dir=$1 --cell_type=$2 --steps=$3
endef

define freeze_graph
	@ echo "freeze_graph"
	@ # echo MODEL_DIR [$1]
	@ # echo OUTPUT_NODE_NAMES $2
	$(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$1/graph.pbtxt \
		--input_checkpoint=$1/model.ckpt-100000 \
		--input_binary=false \
		--output_graph=$1/frozen.pb \
		--output_node_names=$2
endef


# targets
datasets:
	@ mkdir -p $@/tutorial_v1
	@ cd $@ && wget http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz
	@ cd $@ && tar zxvf quickdraw_tutorial_dataset_v1.tar.gz -C tutorial_v1

tutorial-lstm-1k:
	$(call train_model,$(QUICKDRAW_BASE)/$@/models,lstm,1000)
	$(call freeze_graph,$(QUICKDRAW_BASE)/$@/models,dense/BiasAdd)

tutorial-cudnn_lstm-10k:
	$(call train_model,$(QUICKDRAW_BASE)/$@/models,cudnn_lstm,10000)
	$(call freeze_graph,$(QUICKDRAW_BASE)/$@/models,dense/BiasAdd)
