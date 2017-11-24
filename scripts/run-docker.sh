#!/bin/sh
set -e -u

IMAGE_NAME=yumaokao/tensorflow/lite:latest-devel
CONTAINER_NAME=tflite-devel

echo "Cloning tensorflow for container '$CONTAINER_NAME'..."
if [ -d ./tensorflow ]; then
    :
else
    git clone https://github.com/tensorflow/tensorflow
fi

echo "Running container '$CONTAINER_NAME' from image '$IMAGE_NAME'..."

docker start $CONTAINER_NAME > /dev/null 2> /dev/null || {
	echo "Creating new container..."
	docker run \
	       --detach \
	       --name $CONTAINER_NAME \
           --volume $PWD/tensorflow:/home/tflite/tensorflow \
           --publish 8888:8888 \
	       --tty \
	       $IMAGE_NAME
}

if [ "$#" -eq  "0" ]; then
	docker exec --interactive --tty --user tflite $CONTAINER_NAME bash
else
	docker exec --interactive --tty --user tflite $CONTAINER_NAME $@
fi
