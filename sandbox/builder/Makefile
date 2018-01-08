.PHONY: all clean configure
.PHONY: build_tf_pip_gpu build_tf_pip_cpu build_lite_utils

TF_BASE := /home/tflite/tensorflow

all:
	@ echo "all models"

clean:
	@ cd $(TF_BASE) && bazel clean

configure:
	@ cd $(TF_BASE) && ./configure

# build tf
build_tf_pip_gpu:
	@ cd $(TF_BASE) && bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
	@ cd $(TF_BASE) && sudo pip install -U /tmp/tensorflow_pkg/tensorflow-1.5.0rc0-cp27-cp27mu-linux_x86_64.whl


build_tf_pip_cpu:
	@ cd $(TF_BASE) && bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
	@ cd $(TF_BASE) && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
	@ cd $(TF_BASE) && sudo pip install -U /tmp/tensorflow_pkg/tensorflow-1.5.0rc0-cp27-cp27mu-linux_x86_64.whl

build_lite_utils:
	@ cd $(TF_BASE) && bazel build //tensorflow/python/tools:freeze_graph
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/toco:toco
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:dump_tflite
	@ cd $(TF_BASE) && bazel build //tensorflow/contrib/lite/utils:compare_tflite
