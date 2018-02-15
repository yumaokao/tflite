.PHONY: all clean
.PHONY: build_libtoco build_libtoco_test run_libtoco_test

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow

all:
	@ echo "all models"

clean:
	@ rm *.so


# build
build_libtoco:
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:libtoco.so

build_libtoco_test:
	@ rm -f *.so
	@ cp $(TF_BASE)/bazel-bin/tensorflow/contrib/lite/utils/libtoco.so ./
	@ cp $(TF_BASE)/bazel-bin/tensorflow/libtensorflow_framework.so ./
	@ g++ -o libtoco_test libtoco_test.cc libtoco.so libtensorflow_framework.so

run_libtoco_test:
	@ LD_LIBRARY_PATH=. ./libtoco_test
