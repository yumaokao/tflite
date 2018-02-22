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

build_toco_model_test:
	@ rm -rf include
	@ mkdir -p include/tensorflow/contrib/lite/toco/runtime
	@ mkdir -p include/tensorflow/contrib/lite/kernels/internal
	@ mkdir -p include/tensorflow/core/platform/default
	@ cp $(TF_BASE)/tensorflow/contrib/lite/toco/model.h include/tensorflow/contrib/lite/toco
	@ cp $(TF_BASE)/tensorflow/contrib/lite/toco/tooling_util.h include/tensorflow/contrib/lite/toco
	@ sed -i 's/\(.*absl.*\)/\/\/ \1/g' include/tensorflow/contrib/lite/toco/tooling_util.h
	@ sed -i 's/\(#include.*toco_port.h\)/\/\/ \1/g' include/tensorflow/contrib/lite/toco/tooling_util.h
	@ cp $(TF_BASE)/tensorflow/contrib/lite/toco/toco_types.h include/tensorflow/contrib/lite/toco
	@ cp $(TF_BASE)/tensorflow/contrib/lite/toco/model_flags.proto include/tensorflow/contrib/lite/toco
	@ cp $(TF_BASE)/tensorflow/contrib/lite/toco/types.proto include/tensorflow/contrib/lite/toco
	@ cp $(TF_BASE)/tensorflow/contrib/lite/toco/toco_flags.proto include/tensorflow/contrib/lite/toco
	@ protoc --cpp_out=./ include/tensorflow/contrib/lite/toco/types.proto
	@ rm include/tensorflow/contrib/lite/toco/types.pb.cc
	@ protoc -I=./include/ --cpp_out=./include include/tensorflow/contrib/lite/toco/toco_flags.proto
	@ rm include/tensorflow/contrib/lite/toco/toco_flags.pb.cc
	@ protoc -I=./include/ --cpp_out=./include include/tensorflow/contrib/lite/toco/model_flags.proto
	@ rm include/tensorflow/contrib/lite/toco/model_flags.pb.cc
	@ cp $(TF_BASE)/tensorflow/contrib/lite/toco/runtime/types.h include/tensorflow/contrib/lite/toco/runtime
	@ sed -i 's/\(#include.*common.h\)/\/\/ \1/g' include/tensorflow/contrib/lite/toco/runtime/types.h
	@ cp $(TF_BASE)/tensorflow/contrib/lite/kernels/internal/types.h include/tensorflow/contrib/lite/kernels/internal
	@ cp $(TF_BASE)/tensorflow/contrib/lite/kernels/internal/compatibility.h include/tensorflow/contrib/lite/kernels/internal
	@ cp $(TF_BASE)/tensorflow/core/platform/platform.h include/tensorflow/core/platform
	@ cp $(TF_BASE)/tensorflow/core/platform/logging.h include/tensorflow/core/platform
	@ cp $(TF_BASE)/tensorflow/core/platform/macros.h include/tensorflow/core/platform
	@ cp $(TF_BASE)/tensorflow/core/platform/types.h include/tensorflow/core/platform
	@ sed -i 's/\(#include.*bfloat16.h\)/\/\/ \1/g' include/tensorflow/core/platform/types.h
	@ cp $(TF_BASE)/tensorflow/core/platform/default/integral_types.h include/tensorflow/core/platform/default
	@ cp $(TF_BASE)/tensorflow/core/platform/default/logging.h include/tensorflow/core/platform/default
	@ g++ -o toco_model_test toco_model_test.cc libtoco.so libtensorflow_framework.so -I ./include --std=c++11

run_libtoco_test:
	@ LD_LIBRARY_PATH=. ./libtoco_test

run_toco_model_test:
	@ LD_LIBRARY_PATH=. ./toco_model_test
