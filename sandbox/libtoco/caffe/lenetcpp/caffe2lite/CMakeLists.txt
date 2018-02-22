cmake_minimum_required(VERSION 3.0)
project(genlite_converter)

set(CMAKE_CXX_STANDARD 11)

if($ENV{TFLITE_ROOT_PATH})
  set(TFLITE_ROOT_PATH $ENV{TFLITE_ROOT_PATH})
else()
  set(TFLITE_ROOT_PATH /home/tflite)
endif()

add_definitions(-DCPU_ONLY=1)
include_directories("../../caffe/include")
include_directories("./include")
link_directories("../../caffe/build/lib")

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

# genlite
include_directories(FLATBUF_INCLUDE_DIR)
add_executable(genlite genlite.cpp)
add_custom_target(run_genlite DEPENDS genlite
    COMMAND ./genlite model.lite
    COMMAND ${TFLITE_ROOT_PATH}/tensorflow/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite model.lite)
