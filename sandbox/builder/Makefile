.PHONY: all clean configure
.PHONY: build_tf_pip_gpu build_tf_pip_cpu build_lite_utils
.PHONY: install_tf_pip build_install_tf_pip_cpu build_tf_pip_gpu
.PHONY: bazelrc

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

TF_BASE := $(TFLITE_ROOT_PATH)/tensorflow

all:
	@ echo "all models"

clean:
	@ cd $(TF_BASE) && bazel clean

configure:
	@ cd $(TF_BASE) && ./configure

# build tf
build_tf_pip_gpu:
	@ cd $(TF_BASE) && bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

build_tf_pip_cpu:
	@ cd $(TF_BASE) && bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

install_tf_pip:
	@ rm -f /tmp/tensorflow_pkg/*.whl
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
	@ cd $(TF_BASE) && sudo pip install -U /tmp/tensorflow_pkg/tensorflow-*-cp27-cp27mu-linux_x86_64.whl

build_install_tf_pip_gpu: build_tf_pip_gpu install_tf_pip
build_install_tf_pip_cpu: build_tf_pip_cpu install_tf_pip

build_lite_utils:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite

build_libtoco:
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:libtoco.so

# bazelrc
bazelrc:
	@ echo "startup --max_idle_secs=100000000" > ~/.bazelrc
