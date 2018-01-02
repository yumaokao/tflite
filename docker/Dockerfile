# Build with:
# 	docker build -t yumaokao/tensorflow/lite:latest-devel .
# Push to docker hub with:
# 	docker push termux/package-builder
ARG TAG_NAME=latest-devel
FROM gcr.io/tensorflow/tensorflow:${TAG_NAME}
ARG UID
ARG TAG_NAME

# Fix locale to avoid warnings:
ENV LANG C.UTF-8

# Fix bazelrc:
RUN rm /etc/bazel.bazelrc

# Needed for setup:
COPY ./setup-ubuntu.sh /tmp/setup-ubuntu.sh
COPY ./setup-android-sdk.sh /tmp/setup-android-sdk.sh

# Setup needed packages and the Android SDK and NDK:
RUN apt-get update && \
    apt-get -yq upgrade && \
    apt-get install -yq sudo && \
    adduser --disabled-password --uid $UID --shell /bin/bash --gecos "" tflite && \
    echo "tflite ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/tflite && \
    chmod 0440 /etc/sudoers.d/tflite && \
    su - tflite -c /tmp/setup-ubuntu.sh && \
    su - tflite -c /tmp/setup-android-sdk.sh && \
    # Removed unused parts to make a smaller Docker image:
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    # cd /home/tflite/lib/android-ndk/ && \
    # rm -Rf toolchains/mips* && \
    # rm -Rf sources/cxx-stl/gabi++ sources/cxx-stl/llvm-libc++* sources/cxx-stl/system/ sources/cxx-stl/stlport && \
    # cd platforms && ls | grep -v android-21 | xargs rm -Rf && \
    # cd /home/tflite/lib/android-sdk/tools && rm -Rf emulator* lib* proguard templates

WORKDIR /home/tflite
