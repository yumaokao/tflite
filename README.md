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
./scripts/run-docker.sh
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
$ bazel build tensorflow/contrib/lite/toco:toco
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
