#!/usr/bin/env bash
set -e -u

IMAGE_NAME=yumaokao/tensorflow/lite
DOCKER_FILE=./Dockerfile

# CPU
# TAG_NAME=latest-devel
# echo "Building image '$IMAGE_NAME:$TAG_NAME'..."
# docker build -t "$IMAGE_NAME:$TAG_NAME" --build-arg TAG_NAME=$TAG_NAME --build-arg UID=$(id -u) .

# GPU
TAG_NAME=latest-devel-gpu
echo "Building image '$IMAGE_NAME:$TAG_NAME'..."
nvidia-docker build -t "$IMAGE_NAME:$TAG_NAME" --build-arg TAG_NAME=$TAG_NAME --build-arg UID=$(id -u) .
