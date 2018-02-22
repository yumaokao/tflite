cmake_minimum_required(VERSION 3.0)
project(genlite_converter)

set(CMAKE_CXX_STANDARD 11)

if($ENV{TFLITE_ROOT_PATH} STREQUAL "")
  set(TFLITE_ROOT_PATH /home/tflite)
else()
  set(TFLITE_ROOT_PATH $ENV{TFLITE_ROOT_PATH})
endif()

add_definitions(-DCPU_ONLY=1)
include_directories("../caffe/include")
include_directories("./include")
include_directories(FLATBUF_INCLUDE_DIR)
link_directories("../caffe/build/lib")

# find_package(Protobuf REQUIRED)

# setup flatbuffers
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers)
  MESSAGE(FATAL_ERROR "Please clone the flatbuffer source code first")
endif()
# set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUF_SRC_ROOT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers)
set(FLATBUF_INCLUDE_DIR ${FLATBUF_SRC_ROOT_FOLDER}/include)
include_directories(${FLATBUF_INCLUDE_DIR})
# add_subdirectory(${FLATBUF_SRC_ROOT_FOLDER})

# get_lenet_model
add_custom_target(get_lenet_model
    COMMAND cp ../../caffe/examples/mnist/lenet.prototxt ./
    COMMAND cp ../../caffe/examples/mnist/lenet_iter_10000.caffemodel ./)

# genlite
add_executable(genlite genlite.cpp)
add_custom_target(run_genlite DEPENDS genlite
    COMMAND ./genlite model.lite)
add_custom_target(run_dump_tflite DEPENDS genlite
    COMMAND ${TFLITE_ROOT_PATH}/tensorflow/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite model.lite)

# caffe2lite
add_executable(caffe2lite caffe2lite.cpp)
target_link_libraries(caffe2lite caffe boost_system ${PROTOBUF_LIBRARIES})
add_custom_target(run_caffe2lite DEPENDS caffe2lite
    COMMAND ./caffe2lite ./lenet.prototxt ./lenet_iter_10000.caffemodel model.lite
    COMMAND ${TFLITE_ROOT_PATH}/tensorflow/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite model.lite)
