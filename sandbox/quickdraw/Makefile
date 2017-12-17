.PHONY: all clean
.PHONY: datasets
.PHONY: tutorial tutorial-gpu

TF_BASE := /home/tflite/tensorflow
QUICKDRAW_BASE := /home/tflite/sandbox/quickdraw

# functions
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

tutorial:
	# @ python $@/train_model.py \
	  # --training_data=datasets/tutorial_v1/training.tfrecord-?????-of-????? \
	  # --eval_data=datasets/tutorial_v1/eval.tfrecord-?????-of-????? \
	  # --classes_file=datasets/tutorial_v1/training.tfrecord.classes \
	  # --model_dir=$@/models
	$(call freeze_graph,$(QUICKDRAW_BASE)/$@/models,dense/BiasAdd)

tutorial-gpu:
	# @ python $@/train_model.py \
	  # --training_data=datasets/tutorial_v1/training.tfrecord-?????-of-????? \
	  # --eval_data=datasets/tutorial_v1/eval.tfrecord-?????-of-????? \
	  # --classes_file=datasets/tutorial_v1/training.tfrecord.classes \
	  # --cell_type=cudnn_lstm \
	  # --model_dir=$@/models
	$(call freeze_graph,$(QUICKDRAW_BASE)/$@/models,dense/BiasAdd)
