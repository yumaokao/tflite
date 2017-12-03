# Tensorflow Lite Develop Environment

## Prepare Docker
```sh
$ cd docker
$ ./build-docker.sh
# This steps build a docker image from tensorflow:latest-devel
# and add required android sdk and ndk
# and with a normal user tflite (995) for Arch uid starts below 1000
```

## Run Docker
```sh
$ ./scripts/run-docker.sh
# This will also git clone a tensorflow if there is not exist
```

## Prepare WORKSPACE
```sh
android_sdk_repository(
    name = "androidsdk",
    api_level = 23,
    build_tools_version = "26.0.1",
    path = "/home/tflite/lib/android-sdk",
)

android_ndk_repository(
    name="androidndk",
    path="/home/tflite/lib/android-ndk",
    api_level=14)
```

## Build TFLite Demo App
```sh
$ bazel build --cxxopt='--std=c++11' //tensorflow/contrib/lite/java/demo/app/src/main:TfLiteCameraDemo
```

## Build toco
```sh
$ bazel build //tensorflow/contrib/lite/toco:toco
```

## Build freeze_graph
```sh
$ bazel build tensorflow/python/tools:freeze_graph
```

## Build cc_test, py_test
```sh
$ bazel build //tensorflow/contrib/lite:model_test
Target //tensorflow/contrib/lite:model_test up-to-date:
  bazel-bin/tensorflow/contrib/lite/model_test
$ bazel-bin/tensorflow/contrib/lite/model_test

$ bazel build //tensorflow/contrib/quantize:input_to_ops_test
Target //tensorflow/contrib/quantize:input_to_ops_test up-to-date:
  bazel-bin/tensorflow/contrib/quantize/input_to_ops_test
$ bazel-bin/tensorflow/contrib/quantize/input_to_ops_test
```

## Build pip Package
```sh
# CPU
$ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
# GPU
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ ls /tmp/tensorflow_pkg/
tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl

$ sudo pip install -U /tmp/tensorflow_pkg/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
```

# MNIST

## train with saver and summary
```sh
$ cd /home/tflite/sandbox/mnist
$ python ./mnist_deep.py
# or
$ python ./mnist_deep_nodropout.py
test accuracy 0.9921
Model saved in file: /home/tflite/sandbox/mnist/model.ckpt
```

## tensorboard
```sh
$ tensorboard --logdir=/home/tflite/sandbox/mnist/summary
Open http://192.168.10.50:6006
```

## frozen MNIST
```sh
$ bazel-bin/tensorflow/python/tools/freeze_graph \
    --input_graph=/home/tflite/sandbox/mnist/mnist.pb \
    --input_checkpoint=/home/tflite/sandbox/mnist/model.ckpt \
    --input_binary=true --output_graph=/home/tflite/sandbox/mnist/frozen_mnist.pb \
    --output_node_names=loss/Reshape
```

## toco MNIST
```sh
$ bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=/home/tflite/sandbox/mnist/frozen_mnist.pb \
  --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
  --output_file=/tmp/mnist.lite --inference_type=FLOAT \
  --influrence_input_type=FLOAT --input_arrays=reshape/Reshape \
  --output_arrays=fc2/add --input_shapes=1,28,28,1
```

## toco MNIST dummy-quantization
```sh
$ bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=/home/tflite/sandbox/mnist/frozen_mnist.pb \
  --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
  --output_file=/tmp/mnist_quan.lite --inference_type=QUANTIZED_UINT8 \
  --influrence_input_type=QUANTIZED_UINT8 --input_array=reshape/Reshape \
  --output_array=fc2/add --input_shape=1,28,28,1 \
  --default_ranges_min=0 --default_ranges_max=6 \
  --mean_value=127.5 --std_value=127.5
```

## MNIST apks
```sh
# build
$ bazel build --cxxopt='--std=c++11' //tensorflow/contrib/lite/java/demo/app/src/main:TfLiteCameraDemo
$ rm -f ../sandbox/TfLiteCameraDemo.apk
$ cp bazel-bin/tensorflow/contrib/lite/java/demo/app/src/main/TfLiteCameraDemo.apk ../sandbox/
# install
$ adb shell pm uninstall -k com.example.android.tflitecamerademo && adb install -f TfLiteCameraDemo.apk
```


# nvidia-docker

## tensorflow lastest-gpu docker image
```sh
$ nvidia-docker run -it tensorflow/tensorflow:latest-devel-gpu bash
```

## tensorflow lastest-gpu docker image
```sh
$ DEVICE=GPU scripts/run-docker.sh
# This will also git clone a tensorflow if there is not exist
```


# Notes for Tensorflow Lite

## mobilenet
change directory to `tensorflow/tensorflow/contrib/lite`

in `java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java`
```
/** Name of the model file stored in Assets. */
  private static final String MODEL_PATH = "mobilenet_quant_v1_224.tflite";
```

in `java/demo/app/src/main/BUILD`, there is a section
```
android_binary(
    name = "TfLiteCameraDemo",
    srcs = glob(["java/**/*.java"]),
    assets = [
        "@tflite_mobilenet//:labels.txt",
        "@tflite_mobilenet//:mobilenet_quant_v1_224.tflite",
    ],
```

change directory to `tensorflow`

in `tensorflow/workspace.bzl`
```
native.new_http_archive(
      name = "tflite_mobilenet",
      build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
      sha256 = "23f814d1c076bdf03715dfb6cab3713aa4fbdf040fd5448c43196bd2e97a4c1b",
      urls = [
          "https://mirror.bazel.build/storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip",
          "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip",
      ],

```

in `/home/tflite/.cache/bazel/_bazel_tflite/f82e7d13eaeac899986b03b38680d292/external/tflite_mobilenet`
here is where @tflite_mobilenet stores

## dummy-quantization with graph visualize
```sh
$ curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
$ bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.cc \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --default_ranges_min=0 \
  --default_ranges_max=6 \
  --mean_value=127.5 \
  --std_value=127.5 \
  --dump_graphviz=/home/tflite/sandbox/dots

$ cd /home/tflite/sandbox/dots
$ dot -Tpdf -O ./toco_*.dot
```
