cmake_minimum_required(VERSION 3.0)
project(lenetcpp)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-DCPU_ONLY=1)
include_directories("../caffe/include")
link_directories("../caffe/build/lib")

set(SOURCE_FILES lenetcpp.cpp)
add_executable(lenetcpp ${SOURCE_FILES})

find_package(OpenCV REQUIRED)
target_link_libraries(lenetcpp caffe boost_system ${OpenCV_LIBS})

add_custom_target(get_lenet_model
    COMMAND cp -r ../../caffe/examples/mnist/mnist_test_lmdb ./
    COMMAND cp ../../caffe/examples/mnist/lenet_train_test.prototxt ./
    COMMAND cp ../../caffe/examples/mnist/lenet.prototxt ./
    COMMAND cp ../../caffe/examples/mnist/lenet_iter_10000.caffemodel ./
    COMMAND sed -e 's/examples.mnist.//' lenet_train_test.prototxt > lenet_train_test_local.prototxt)

add_custom_target(run_lenetcpp DEPENDS lenetcpp
    COMMAND ./lenetcpp lenet.prototxt lenet_iter_10000.caffemodel 1)
