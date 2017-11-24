#!/usr/bin/env bash
set -e -u

IMAGE_NAME=yumaokao/tensorflow/lite:latest-devel
DOCKER_FILE=./Dockerfile

echo "Building image '$IMAGE_NAME'..."

docker build -t $IMAGE_NAME .
