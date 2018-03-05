.PHONY: all clean
.PHONY: clone_flatbuffers
.PHONY: build_genlite run_genlite run_dump_tflite

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow

all:
	@ echo "all models"

clean:
	@ rm -rf build

# clone
clone_flatbuffers:
	@ [ -d flatbuffers ] || git clone https://github.com/google/flatbuffers.git

# build
build_genlite:
	@ [ -d build ] || mkdir build
	@ cd build && cmake ../
	@ cd build && make genlite

run_genlite:
	@ [ -d build ] || mkdir build
	@ cd build && cmake ../
	@ cd build && make run_genlite

run_dump_tflite:
	@ cd build && $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite model.lite
