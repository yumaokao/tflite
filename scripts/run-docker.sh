#!/bin/sh
set -e -u

IMAGE_NAME=yumaokao/tensorflow/lite:latest-devel
CONTAINER_NAME=tflite-devel

echo "Running container '$CONTAINER_NAME' from image '$IMAGE_NAME'..."

docker start $CONTAINER_NAME > /dev/null 2> /dev/null || {
	echo "Creating new container..."
	docker run \
	       --detach \
	       --name $CONTAINER_NAME \
           --publish 8888:8888 \
	       --tty \
	       $IMAGE_NAME
}

if [ "$#" -eq  "0" ]; then
	docker exec --interactive --tty $CONTAINER_NAME bash
else
	docker exec --interactive --tty $CONTAINER_NAME $@
fi
