.PHONY: all clean configure
.PHONY: build_tf_pip_gpu build_tf_pip_cpu build_lite_utils
.PHONY: install_tf_pip build_install_tf_pip_cpu build_tf_pip_gpu
.PHONY: bazelrc install_bazel

ifeq ($(TFLITE_ROOT_PATH),)
TFLITE_ROOT_PATH := /home/tflite
endif

ifeq ($(TF_PACKAGE_PATH),)
TF_PACKAGE_PATH := /tmp/tensorflow_pkg
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
	@ rm -f $(TF_PACKAGE_PATH)/*.whl
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/pip_package/build_pip_package $(TF_PACKAGE_PATH)
	@ cd $(TF_BASE) && sudo pip install -U $(TF_PACKAGE_PATH)/tensorflow-*-cp27-cp27mu-linux_x86_64.whl

install_tf_pip_local:
	@ rm -f $(TF_PACKAGE_PATH)/*.whl
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/pip_package/build_pip_package $(TF_PACKAGE_PATH)
	@ cd $(TF_BASE) && pip install -U $(TF_PACKAGE_PATH)/tensorflow-*-cp27-cp27mu-linux_x86_64.whl

build_install_tf_pip_gpu: build_tf_pip_gpu install_tf_pip
build_install_tf_pip_cpu: build_tf_pip_cpu install_tf_pip

build_lite_utils:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:run_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:summarize_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/tools/graph_transforms:transform_graph

build_libtoco:
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:libtoco.so

# bazelrc
bazelrc:
	@ echo "startup --max_idle_secs=100000000" > ~/.bazelrc

install_bazel:
	@ wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel-0.11.1-installer-linux-x86_64.sh
	@ sudo chmod +x bazel-0.11.1-installer-linux-x86_64.sh
	@ sudo ./bazel-0.11.1-installer-linux-x86_64.sh
