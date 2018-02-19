cmake_minimum_required(VERSION 3.0)
project(lenetcpp)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-DCPU_ONLY=1)
include_directories("../caffe/include")
link_directories("../caffe/build/lib")

find_package(OpenCV REQUIRED)

add_executable(lenetinf lenetinf.cpp)
target_link_libraries(lenetinf caffe boost_system ${OpenCV_LIBS})

add_custom_target(get_lenet_model
    COMMAND cp ../../caffe/examples/mnist/lenet.prototxt ./
    COMMAND cp ../../caffe/examples/mnist/lenet_iter_10000.caffemodel ./)

add_custom_target(run_lenetinf DEPENDS lenetinf
    COMMAND ./lenetinf lenet.prototxt lenet_iter_10000.caffemodel 1)
