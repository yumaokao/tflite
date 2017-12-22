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
	@ # echo STEPS $3
	$(TF_BASE)/bazel-bin/tensorflow/python/tools/freeze_graph \
		--input_graph=$1/graph.pbtxt \
		--input_checkpoint=$1/model.ckpt-$3 \
		--input_binary=false \
		--output_graph=$1/frozen.pb \
		--output_node_names=$2
endef

define toco
	@ echo "toco"
	@ mkdir -p $1/dots
	@ mkdir -p $1/export
	@ $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/toco/toco \
		--input_file=$1/frozen.pb \
		--input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
		--output_file=$1/export/model.lite \
		--inference_type=FLOAT \
		--inference_input_type=FLOAT --input_arrays=$3 \
		--output_arrays=$4 --input_shapes=8,16,3 \
		--dump_graphviz=$1/dots $5
	# echo "cd $1/dots && dot -Tpdf -O ./toco_*.dot"
endef


# targets
datasets:
	@ mkdir -p $@/tutorial_v1
	@ cd $@ && wget http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz
	@ cd $@ && tar zxvf quickdraw_tutorial_dataset_v1.tar.gz -C tutorial_v1

clean:
	@ rm -rf tutorial-*

tutorial-lstm-1k:
	$(call train_model,$(QUICKDRAW_BASE)/$@/models,lstm,1000)
	$(call freeze_graph,$(QUICKDRAW_BASE)/$@/models,conv1d_0/BiasAdd,1000)
	$(call toco,$(QUICKDRAW_BASE)/$@/models,frozen.pb,Reshape,conv1d_0/BiasAdd,)

tutorial-cudnn_lstm-10k:
	$(call train_model,$(QUICKDRAW_BASE)/$@/models,cudnn_lstm,10000)
	$(call freeze_graph,$(QUICKDRAW_BASE)/$@/models,dense/BiasAdd)
	# $(call toco,$(OPS_BASE)/$@/models,frozen.pb,Reshape,dense/BiasAdd,)
