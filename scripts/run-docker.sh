#!/bin/sh
set -e -u

DEVICE=${DEVICE:=CPU}
TAG_NAME=latest-devel
DOCKER=docker
[ $DEVICE == "GPU" ] && TAG_NAME=latest-devel-gpu
[ $DEVICE == "GPU" ] && DOCKER=nvidia-docker
DATASETS_DIR=${DATASETS_DIR:=/home/yumaokao/mnts/sda2/yumaokao/datasets}
[ -d $DATASETS_DIR ] || DATASETS_DIR=$PWD/datasets

IMAGE_NAME=yumaokao/tensorflow/lite:$TAG_NAME
CONTAINER_NAME=tflite-$TAG_NAME

echo "Cloning tensorflow for container '$CONTAINER_NAME'..."
if [ -d ./tensorflow ]; then
    :
else
    git clone https://yumaokao@github.com/yumaokao/tensorflow
    cd tensorflow && git checkout origin/lite-utils -b lite-utils && cd -
fi
if [ -d ./models ]; then
    :
else
    git clone https://yumaokao@github.com/yumaokao/models
fi

echo "Running container '$CONTAINER_NAME' from image '$IMAGE_NAME' with '$DOCKER'..."

$DOCKER start $CONTAINER_NAME > /dev/null 2> /dev/null || {
	echo "Creating new container..."
	$DOCKER run \
	       --detach \
	       --name $CONTAINER_NAME \
           --volume $PWD/tensorflow:/home/tflite/tensorflow \
           --volume $PWD/sandbox:/home/tflite/sandbox \
           --volume $PWD/models:/home/tflite/models \
           --volume $DATASETS_DIR:/home/tflite/datasets \
	       --tty \
	       $IMAGE_NAME
}

if [ "$#" -eq  "0" ]; then
	$DOCKER exec --interactive --tty --user tflite $CONTAINER_NAME bash
else
	$DOCKER exec --interactive --tty --user tflite $CONTAINER_NAME $@
fi
