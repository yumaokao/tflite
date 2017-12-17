.PHONY: all clean
.PHONY: datasets
.PHONY: tutorial tutorial-gpu

TF_BASE := /home/tflite/tensorflow
QUICKDRAW_BASE := /home/tflite/sandbox/quickdraw

datasets:
	@ mkdir -p $@/tutorial_v1
	@ cd $@ && wget http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz
	@ cd $@ && tar zxvf quickdraw_tutorial_dataset_v1.tar.gz -C tutorial_v1

tutorial:
	@ python $@/train_model.py \
	  --training_data=datasets/tutorial_v1/training.tfrecord-?????-of-????? \
	  --eval_data=datasets/tutorial_v1/eval.tfrecord-?????-of-????? \
	  --classes_file=datasets/tutorial_v1/training.tfrecord.classes \
	  --model_dir=$@/models

tutorial-gpu:
	@ python $@/train_model.py \
	  --training_data=datasets/tutorial_v1/training.tfrecord-?????-of-????? \
	  --eval_data=datasets/tutorial_v1/eval.tfrecord-?????-of-????? \
	  --classes_file=datasets/tutorial_v1/training.tfrecord.classes \
	  --cell_type=cudnn_lstm \
	  --model_dir=$@/models
